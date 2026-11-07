A `user` turn of `client.generation` can hold images and PDF documents:
`Turn.user([...])` takes a list of `TextPart`, `ImagePart` and
`DocumentPart`, each image or document given as bytes (sent inline in
base64) or as a URL. `await ImagePart.load(path)` and
`await DocumentPart.load(path)` read a file in a worker thread. OpenAI
gets `input_text`, `input_image` and `input_file` items, Anthropic
`text`, `image` and `document` blocks; parts in an `assistant` turn
raise `ValidationError`.
