"""CPU-only reproduction of Torch FD transfer blocked by a native GIL owner.

Compile the actual GatedMLP binding line around a bounded native wait, then
exercise a real spawned worker and Torch file_descriptor resource_sharer.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
CPP = r'''
#include <pybind11/pybind11.h>
#include <poll.h>
#include <unistd.h>
namespace py=pybind11;
class BC_GatedMLP {
public:
    bool run_bszN(int begin_fd, int ack_fd) {
        // Connection.send_bytes framing: tell the child to deserialize only
        // after this native call has entered its held/released GIL scope.
        const char begin[] = {0,0,0,1,'g'};
        if (write(begin_fd,begin,5)!=5) return false;
        pollfd descriptor{ack_fd,POLLIN,0};
        return poll(&descriptor,1,1000)>0;
    }
};
PYBIND11_MODULE(MODULE_NAME, m) {
    py::class_<BC_GatedMLP>(m,"BC_GatedMLP",py::module_local()).def(py::init<>())
BINDING_LINE
}
'''


def worker(connection, signal):
    connection.send('ready')
    signal.recv_bytes()
    tensor = connection.recv()
    connection.send(tensor.numel())


class GILHandshakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        compiler = shutil.which('c++')
        if not compiler:
            raise unittest.SkipTest('C++ compiler required')
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        line = '.def("run_bszN", &BC_GatedMLP::run_bszN);'
        header = (ROOT / 'engine-src/exllamav3/exllamav3_ext/libtorch/mlp_bc.h').read_text()
        if header.count(line) != 1:
            raise AssertionError('unexpected original GatedMLP binding')
        guarded = line.replace(');', ', py::call_guard<py::gil_scoped_release>());')
        patch = (Path(__file__).parent / '0002-release-gil-during-gated-mlp.patch').read_text()
        if '\n+' + guarded not in patch:
            raise AssertionError('candidate patch does not supply the tested call guard')
        cls.modules = []
        for name, binding in [('gated_probe_held', line), ('gated_probe_released', guarded)]:
            cpp = root / (name+'.cpp')
            cpp.write_text(CPP.replace('MODULE_NAME',name).replace('BINDING_LINE',binding))
            so = root / (name+sysconfig.get_config_var('EXT_SUFFIX'))
            include = Path(torch.__file__).parent / 'include'
            subprocess.run([compiler,'-shared','-fPIC','-O2','-std=c++17',
                            '-I'+str(include),'-I'+sysconfig.get_path('include'),
                            str(cpp),'-o',str(so)],check=True,capture_output=True)
            spec=importlib.util.spec_from_file_location(name,so)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            cls.modules.append(module)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_shared_tensor_handshake_requires_released_gil(self):
        ctx=self.torch.multiprocessing.get_context('spawn')
        self.torch.multiprocessing.set_sharing_strategy('file_descriptor')
        for module,expected in zip(self.modules,[False,True]):
            with self.subTest(binding=module.__name__):
                parent,child=ctx.Pipe()
                receive,send=ctx.Pipe(duplex=False)
                process=ctx.Process(target=worker,args=(child,receive))
                process.start()
                try:
                    self.assertTrue(parent.poll(15),'worker did not become ready')
                    self.assertEqual(parent.recv(),'ready')
                    parent.send(self.torch.zeros(1024))
                    self.assertIs(module.BC_GatedMLP().run_bszN(send.fileno(),parent.fileno()),expected)
                    self.assertTrue(parent.poll(5),'worker failed after native wait')
                    self.assertEqual(parent.recv(),1024)
                    process.join(5)
                    self.assertEqual(process.exitcode,0)
                finally:
                    if process.is_alive():process.terminate();process.join(5)
                    for connection in [parent,child,receive,send]:connection.close()


if __name__ == '__main__':
    unittest.main()
