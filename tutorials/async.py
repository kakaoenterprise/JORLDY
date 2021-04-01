import ray
import asyncio
ray.init()

@ray.remote
class AsyncWorker:
    async def do_work(self):
        print("Started")
        await asyncio.sleep(0.5) # simulate network I/O
        print("Ended")
        return "Done!"

worker = AsyncWorker.remote()

async def main():
    ray_future = worker.do_work.remote()
    result = await ray_future
    assert result == "Done!"

    many_task_futures = [worker.do_work.remote() for _ in range(20)]
    result = await asyncio.gather(*many_task_futures)

asyncio.get_event_loop().run_until_complete(main())