import unittest
from unittest.mock import AsyncMock, Mock, patch
from switcher import Services


class FakeServices(Services):
    def __init__(self, free=True): self.events = []; self.free = free
    async def command(self, argv, **kw): self.events.append(('command', argv, kw))
    async def free_gpus(self):
        self.events.append(('gpu-free',))
        if not self.free: raise RuntimeError('still occupied')
    async def ready(self, target): self.events.append(('ready', target))


class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_start_waits_for_glm_stop_and_free_gpus(self):
        b = FakeServices(); await b.switch('qwen')
        self.assertEqual(b.events[0][1], ['systemctl','--user','stop','glm53-b2.service'])
        self.assertTrue(b.events[0][2]['allow_failure'])
        self.assertEqual(b.events[1], ('gpu-free',))
        self.assertEqual(b.events[2][1], ['sudo','-n','systemctl','start','qwen-flash-vllm.service'])
        self.assertEqual(b.events[3], ('ready','qwen'))

    async def test_glm_start_waits_for_qwen_stop_and_free_gpus(self):
        b = FakeServices(); await b.switch('glm')
        self.assertEqual(b.events[0][1], ['sudo','-n','systemctl','stop','qwen-flash-vllm.service'])
        self.assertEqual(b.events[1], ('gpu-free',))
        self.assertEqual(b.events[2][1], ['systemctl','--user','start','glm53-b2.service'])

    async def test_occupied_gpu_refuses_new_owner(self):
        for target in ('glm','qwen'):
            b = FakeServices(False)
            with self.assertRaises(RuntimeError): await b.switch(target)
            self.assertEqual(len(b.events), 2)
            self.assertNotIn('start', b.events[0][1])

    async def test_hung_gpu_query_is_killed_and_refuses_switch(self):
        process = Mock(communicate=AsyncMock(side_effect=TimeoutError), wait=AsyncMock())
        backend = Services.__new__(Services)
        with patch('switcher.asyncio.create_subprocess_exec', AsyncMock(return_value=process)):
            with self.assertRaisesRegex(RuntimeError, 'ownership query timed out'):
                await backend.free_gpus()
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()


if __name__ == '__main__': unittest.main()
