# Answer : For UTF-8, if we are trying to represent some character in a single byte, then the highest bit of this byte should be 0, so we can use bytes whose highest bit is one and force them to be unable to represent two-byte characters. For example (0xFF 0xFF)

utf_encoded = bytes([255, 255])
# print(utf_encoded.decode("utf-8"))
print(utf_encoded.decode("utf-16"))
# Sadly, it can be translated into utf-16.