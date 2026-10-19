"""Compile the actual native headroom guard against a CPU CUDA boundary.

This checks decisions and sticky fallback, not CUDA numerical equivalence.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREAMBLE = r'''
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
struct Graph { bool disabled=false, ready=false, ready_to_record=true; };
int queries=0, status=0;
size_t available=0;
int cudaMemGetInfo(size_t* free, size_t* total) {
    ++queries; *free=available; *total=64ULL*1024*1024*1024; return status;
}
#define AT_CUDA_CHECK(code) if ((code) != 0) throw std::runtime_error("cuda error")
#define TORCH_CHECK(condition, message) if (!(condition)) throw std::runtime_error(message)
'''
MAIN = r'''
int main(int argc, char** argv) {
    Graph graph;
    available=std::stoull(argv[1])*1024*1024;
    std::string mode=argc>2 ? argv[2] : "";
    graph.ready=mode=="ready";
    graph.ready_to_record=mode!="first";
    status=mode=="error" ? 9 : 0;
    try {
        guard_kda_graph_capture(graph);
        guard_kda_graph_capture(graph);
    } catch (...) { return 8; }
    printf("%d %d\n", graph.disabled, queries);
}
'''


class HeadroomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            raise unittest.SkipTest('C++ compiler required')
        cls.tmp = tempfile.TemporaryDirectory()
        directory = Path(cls.tmp.name)
        rel = Path('exllamav3/exllamav3_ext/libtorch/gated_delta_net.cpp')
        source = directory / rel
        source.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'engine-src' / rel, source)
        subprocess.run(['git', 'apply', str(HERE / '0001-kda-graph-headroom.patch')],
                       cwd=directory, check=True, capture_output=True)
        text = source.read_text()
        method = text[text.index('static void guard_kda_graph_capture'):text.index('at::Tensor BC_GatedDeltaNet::run_bsz1_a')]
        cpp = directory / 'guard.cpp'
        cpp.write_text(PREAMBLE + method + MAIN)
        cls.binary = directory / 'guard'
        subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_guard(self, floor, free, mode=''):
        env = os.environ.copy()
        env['EXL3_KDA_GRAPH_MIN_FREE_MB'] = floor
        return subprocess.run([str(self.binary), str(free), mode], env=env, capture_output=True, text=True)

    def test_low_headroom_uses_sticky_eager_fallback_once(self):
        r = self.run_guard('1024', 643)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, '1 1'))
        self.assertEqual(r.stderr.count('new graph slot remains eager'), 1)

    def test_exact_floor_and_more_preserve_capture_eligibility(self):
        for free in (1024, 2048):
            r = self.run_guard('1024', free)
            self.assertEqual((r.returncode, r.stdout.strip(), r.stderr), (0, '0 2', ''))

    def test_existing_graph_and_first_eager_run_do_not_query_memory(self):
        for mode in ('ready', 'first'):
            r = self.run_guard('1024', 0, mode)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, '0 0'))

    def test_disabled_policy_preserves_previous_behavior(self):
        r = self.run_guard('0', 0)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, '0 0'))

    def test_invalid_policy_and_prior_cuda_errors_surface(self):
        for value in ('', '-1', 'oops', '1024x', '1048577', '99999999999999999999999'):
            self.assertEqual(self.run_guard(value, 2048).returncode, 8)
        self.assertEqual(self.run_guard('1024', 2048, 'error').returncode, 8)


if __name__ == '__main__':
    unittest.main()
