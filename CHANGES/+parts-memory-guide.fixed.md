The generation guide gives the measured peak memory of a request with a
loaded file: about 15 times the file's size with the default orjson
encoder, which briefly holds a buffer far larger than its result for a
long string, and about 4 times with `json_dumps=json.dumps`.
