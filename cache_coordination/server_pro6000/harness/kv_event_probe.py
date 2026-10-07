"""Subscribe to vLLM KV cache events and report what the engine tells us.

Wire format (vllm/distributed/kv_events.py ZmqEventPublisher):
    multipart -> (topic_bytes, seq_bytes_be8, msgpack(KVEventBatch))
The publisher binds when the endpoint contains "*", so we connect.
Events are msgspec Structs with tag=True, so the msgpack payload decodes to
[ts, [[tag, ...fields], ...], data_parallel_rank].
"""

import argparse
import collections
import sys
import time

import msgspec
import zmq

MEDIA = ("GPU", "CPU", "FS", "OBJ")
UNKNOWN = "?"


def medium_of(ev):
    for f in ev[1:]:
        if f in MEDIA:
            return f
    return UNKNOWN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    ap.add_argument("--topic", default="")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(args.endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    print("subscribed to %s topic=%r" % (args.endpoint, args.topic), flush=True)

    dec = msgspec.msgpack.Decoder()
    kinds = collections.Counter()
    media = collections.Counter()
    blocks = collections.Counter()
    shown = 0
    deadline = time.time() + args.seconds

    while time.time() < deadline:
        if not sub.poll(500):
            continue
        _topic, _seq, payload = sub.recv_multipart()
        batch = dec.decode(payload)
        for ev in batch[1]:
            if not isinstance(ev, list):
                kinds[str(ev)] += 1
                continue
            tag = ev[0]
            kinds[tag] += 1
            hashes = ev[1] if len(ev) > 1 and isinstance(ev[1], list) else []
            if tag == "BlockStored":
                blocks[tag] += len(hashes)
                media[medium_of(ev)] += len(hashes)
                if shown < 4:
                    tok = ev[3] if len(ev) > 3 and isinstance(ev[3], list) else []
                    print(
                        "  BlockStored: n_hashes=%d parent=%.24s n_token_ids=%d "
                        "block_size=%s medium=%s"
                        % (len(hashes), str(ev[2]), len(tok),
                           ev[4] if len(ev) > 4 else UNKNOWN, medium_of(ev)),
                        flush=True,
                    )
                    shown += 1
            elif tag == "BlockRemoved":
                blocks[tag] += len(hashes)
                media["-" + medium_of(ev)] += len(hashes)

    print("")
    print("event kinds : %s" % dict(kinds))
    print("blocks      : %s" % dict(blocks))
    print("by medium   : %s" % dict(media))
    if not media.get("CPU") and not media.get("-CPU"):
        print(
            "WARNING: no CPU-medium events -> L2 tier is not reporting "
            "(check self_describing_kv_events / cpu_bytes_to_use)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
