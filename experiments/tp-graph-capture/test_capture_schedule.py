"""Execute the actual capture_begin method against a bounded CPU CUDA model.

This reproduces the synchronization dependency. It does not qualify GPU code.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREAMBLE = r'''
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
using cudaStream_t = void*;
constexpr int cudaStreamNonBlocking=1, cudaStreamCaptureModeThreadLocal=1;
std::atomic<bool> next_collective_posted=false, peer_completed=false;
int pending_error=0;
#define TORCH_CHECK(condition, message) if (!(condition)) throw std::runtime_error(message)
#define AT_CUDA_CHECK(code) if ((code) != 0) throw std::runtime_error("cuda error")
namespace at { namespace cuda { void* getCurrentCUDABlasHandle() { return nullptr; } } }
int cudaGetLastError() { return pending_error; }
int cudaDeviceSynchronize() {
    auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(100);
    while (!peer_completed) {
        if (std::chrono::steady_clock::now()>deadline) return 1;
        std::this_thread::yield();
    }
    return 0;
}
int cudaStreamCreateWithFlags(cudaStream_t* stream, int) { *stream=(void*)1; return 0; }
int cudaStreamBeginCapture(cudaStream_t, int) { return 0; }
struct Graph {
    cudaStream_t capture_stream=nullptr;
    bool capture_active=false;
    cudaStream_t capture_begin();
    void capture_abort() { capture_stream=nullptr; capture_active=false; }
};
'''
MAIN = r'''
int main(int argc, char**) {
    Graph graph;
    if (argc>1) {
        pending_error=9;
        try { graph.capture_begin(); return 5; }
        catch (...) { return graph.capture_active ? 6 : 0; }
    }
    // A queued peer collective awaits this host's next submission. Device-wide
    // synchronization cannot complete until that submission happens.
    std::thread peer([] {
        while (!next_collective_posted) std::this_thread::yield();
        peer_completed=true;
    });
    bool returned=false;
    try { graph.capture_begin(); returned=true; } catch (...) {}
    next_collective_posted=true;
    peer.join();
    return returned && graph.capture_active ? 0 : 7;
}
'''


class CaptureScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("a C++ compiler is required")
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        original = (ROOT / "engine-src/exllamav3/exllamav3_ext/graph.cu").read_text()
        source = cls.directory / "exllamav3/exllamav3_ext/graph.cu"
        source.parent.mkdir(parents=True)
        source.write_text(original)
        subprocess.run(["git", "apply", str(HERE / "0001-nonblocking-graph-capture.patch")],
                       cwd=cls.directory, check=True, capture_output=True)
        patched = source.read_text()
        cls.executables = []
        for name, text in [("original", original), ("patched", patched)]:
            method = text[text.index("cudaStream_t Graph::capture_begin()"):text.index("void Graph::capture_abort()")]
            cpp = cls.directory / (name + ".cpp")
            cpp.write_text(PREAMBLE + method + MAIN)
            executable = cls.directory / name
            subprocess.run([compiler, "-std=c++17", "-pthread", str(cpp), "-o", str(executable)],
                           check=True, capture_output=True)
            cls.executables.append(executable)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_original_waits_on_peer_while_candidate_allows_submission(self):
        original, patched = self.executables
        self.assertEqual(subprocess.run([str(original)], timeout=2).returncode, 7)
        self.assertEqual(subprocess.run([str(patched)], timeout=2).returncode, 0)

    def test_candidate_surfaces_pending_launch_error_before_capture(self):
        self.assertEqual(subprocess.run([str(self.executables[1]), "error"], timeout=2).returncode, 0)


if __name__ == "__main__":
    unittest.main()
