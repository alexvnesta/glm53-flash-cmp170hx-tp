"""Actual Starlette/AnyIO ASGI calls; no Torch, model or GPU imports."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
import sys
import threading
import unittest

import anyio
from glm_response_cleanup import OwnedJobCleanup, OwnedStreamingResponse


SCOPE={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},
       'http_version':'1.1','method':'GET','path':'/','headers':[]}


def leaves(error):
    if isinstance(error,BaseExceptionGroup):
        return [e for child in error.exceptions for e in leaves(child)]
    return [error]


class Lock(asyncio.Lock):
    def __init__(self):super().__init__();self.release_calls=0;self.check=None
    def release(self):
        if self.check:self.check()
        self.release_calls+=1
        super().release()


class Job:
    def __init__(self,lock,error=None):
        self.lock=lock;self.error=error;self.calls=0;self.finished=False
        self.started=asyncio.Event();self.allow_finish=asyncio.Event()
    async def cancel(self):
        self.calls+=1
        assert self.lock.locked()
        self.started.set()
        await self.allow_finish.wait()
        assert self.lock.locked()
        self.finished=True
        if self.error:raise self.error


class ResponseContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.lock=Lock();await self.lock.acquire();self.job=Job(self.lock)
        self.lock.check=lambda:self.assertTrue(self.job.finished)
        self.cleanup=OwnedJobCleanup(self.job.cancel,self.lock.release)
        self.body_entered=False;self.messages=[]
        self.never=asyncio.Event()

    async def receive(self):
        await self.never.wait()
        return {'type':'http.disconnect'}

    async def send(self,message):self.messages.append(message)

    async def body(self):
        self.body_entered=True
        try:
            yield 'data: first\n\n'
            yield 'data: [DONE]\n\n'
        finally:
            await self.cleanup.close()

    def response(self,content=None):
        return OwnedStreamingResponse(self.body() if content is None else content,
                                      cleanup=self.cleanup,media_type='text/event-stream')

    async def wait(self,event):await asyncio.wait_for(event.wait(),2)

    def assert_released_once(self):
        self.assertFalse(self.lock.locked());self.assertEqual(self.lock.release_calls,1)
        self.assertEqual(self.job.calls,1);self.assertTrue(self.cleanup.released)

    async def test_normal_stream_iterator_and_outer_share_cleanup(self):
        self.job.allow_finish.set()
        await self.response()(SCOPE,self.receive,self.send)
        self.assertTrue(self.body_entered);self.assert_released_once()
        bodies=[m['body'] for m in self.messages if m['type']=='http.response.body']
        self.assertEqual(bodies,[b'data: first\n\n',b'data: [DONE]\n\n',b''])
        await self.cleanup.close();self.assert_released_once()

    async def test_header_send_failure_before_body_entry_drains_cleanup(self):
        primary=RuntimeError('header send failed')
        async def fail(message):raise primary
        task=asyncio.create_task(self.response()(SCOPE,self.receive,fail))
        await self.wait(self.job.started)
        self.assertFalse(self.body_entered);self.assertTrue(self.lock.locked())
        self.assertFalse(task.done());self.job.allow_finish.set()
        with self.assertRaises(BaseException) as caught:await task
        self.assertIn(primary,leaves(caught.exception));self.assert_released_once()

    async def test_cancel_before_body_entry_with_repeated_task_cancel(self):
        header_entered=asyncio.Event()
        async def blocked_header(message):
            header_entered.set();await self.never.wait()
        task=asyncio.create_task(self.response()(SCOPE,self.receive,blocked_header))
        await self.wait(header_entered);task.cancel();await self.wait(self.job.started)
        task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done());self.assertFalse(self.body_entered)
        self.assertTrue(self.lock.locked());self.job.allow_finish.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assert_released_once()

    async def test_disconnect_before_iterator_entry(self):
        async def disconnected():return {'type':'http.disconnect'}
        async def blocked_send(message):await self.never.wait()
        task=asyncio.create_task(self.response()(SCOPE,disconnected,blocked_send))
        await self.wait(self.job.started)
        self.assertFalse(self.body_entered);self.assertTrue(self.lock.locked())
        self.job.allow_finish.set();await task;self.assert_released_once()

    async def test_anyio_level_cancellation_is_shielded(self):
        started=asyncio.Event();holder={}
        async def blocked_header(message):started.set();await self.never.wait()
        async def invoke():
            with anyio.CancelScope() as scope:
                holder['scope']=scope
                await self.response()(SCOPE,self.receive,blocked_header)
        task=asyncio.create_task(invoke());await self.wait(started)
        holder['scope'].cancel();await self.wait(self.job.started)
        holder['scope'].cancel();await asyncio.sleep(0)
        self.assertFalse(task.done());self.assertTrue(self.lock.locked())
        self.job.allow_finish.set();await task;self.assert_released_once()

    async def test_job_cancel_exception_is_cached_release_once(self):
        error=ValueError('job cancellation failed');self.job.error=error
        self.job.allow_finish.set()
        for _ in range(2):
            with self.assertRaises(ValueError) as caught:await self.cleanup.close()
            self.assertIs(caught.exception,error)
        self.assertIs(self.cleanup.cancel_error,error);self.assert_released_once()

    async def test_header_failure_preserved_when_job_cancel_also_raises(self):
        primary=RuntimeError('header failed');error=ValueError('cancel failed')
        self.job.error=error;self.job.allow_finish.set()
        async def fail(message):raise primary
        with self.assertRaises(BaseException) as caught:
            await self.response()(SCOPE,self.receive,fail)
        self.assertIn(primary,leaves(caught.exception))
        self.assertTrue(any('Owned job cleanup also failed' in n
                            for n in getattr(caught.exception,'__notes__',())))
        self.assertIs(self.cleanup.cancel_error,error);self.assert_released_once()

    async def test_concurrent_cleanup_callers_cancel_job_only_once(self):
        a=asyncio.create_task(self.cleanup.close());b=asyncio.create_task(self.cleanup.close())
        await self.wait(self.job.started);a.cancel();await asyncio.sleep(0)
        self.assertFalse(a.done());self.assertFalse(b.done());self.assertEqual(self.job.calls,1)
        self.job.allow_finish.set()
        with self.assertRaises(asyncio.CancelledError):await a
        await b;self.assert_released_once()

    async def test_body_send_failure_outer_cleanup_then_iterator_finally_idempotent(self):
        self.job.allow_finish.set();body=self.body();count=0
        async def fail_body(message):
            nonlocal count
            count+=1
            if message['type']=='http.response.body':raise RuntimeError('body send failed')
        with self.assertRaises(BaseException) as caught:
            await self.response(body)(SCOPE,self.receive,fail_body)
        self.assertTrue(any(str(e)=='body send failed' for e in leaves(caught.exception)))
        self.assert_released_once()
        await body.aclose();self.assert_released_once()

    async def test_cancel_failure_propagates_from_outer_when_body_never_entered(self):
        error=ValueError('cancel failed');self.job.error=error;self.job.allow_finish.set()
        async def disconnected():return {'type':'http.disconnect'}
        async def blocked_send(message):await self.never.wait()
        with self.assertRaises(ValueError) as caught:
            await self.response()(SCOPE,disconnected,blocked_send)
        self.assertIs(caught.exception,error);self.assert_released_once()

    async def test_blocked_executor_cancel_is_not_cancelled_by_callers(self):
        pool=ThreadPoolExecutor(max_workers=1);entered=threading.Event();release=threading.Event()
        record=[];main=threading.get_ident()
        def native_cancel():
            entered.set()
            if not release.wait(3):raise TimeoutError('test executor stalled')
            record.append(threading.get_ident())
        async def cancel():
            self.job.calls+=1;self.job.started.set()
            await asyncio.get_running_loop().run_in_executor(pool,native_cancel)
            self.job.finished=True
        self.cleanup=OwnedJobCleanup(cancel,self.lock.release)
        task=asyncio.create_task(self.cleanup.close())
        try:
            await self.wait(self.job.started)
            for _ in range(3):task.cancel();await asyncio.sleep(0)
            self.assertFalse(task.done());self.assertTrue(self.lock.locked())
            release.set()
            with self.assertRaises(asyncio.CancelledError):await task
            self.assert_released_once();self.assertEqual(len(record),1);self.assertNotEqual(record[0],main)
        finally:release.set();await asyncio.to_thread(pool.shutdown,wait=True)

    async def test_synchronous_callback_failure_still_releases_once(self):
        error=ValueError('callback failed before returning coroutine')
        def fail():self.job.calls+=1;self.job.finished=True;raise error
        self.cleanup=OwnedJobCleanup(fail,self.lock.release)
        with self.assertRaises(ValueError):await self.cleanup.close()
        with self.assertRaises(ValueError):await self.cleanup.close()
        self.assert_released_once()


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResponseContracts))
    print(json.dumps({'tests':result.testsRun,'passed':result.wasSuccessful(),
                      'python':sys.version.split()[0],'starlette':version('starlette'),
                      'anyio':version('anyio'),'torch_imported':'torch' in sys.modules}))
    raise SystemExit(not result.wasSuccessful())
