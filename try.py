# You should type python in cmd to see what happen.

# 2.1
print('='*25 + "2.1" + '='*25)

chr(0)
print(chr(0))
"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")

print('='*53 + '\n')

# 2.2
print('='*25 + "2.2.1" + '='*25)

test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(type(utf8_encoded))
# Get the byte values for the encoded string (integers from 0 to 255).
print(list(utf8_encoded))
# One byte does not necessarily correspond to one Unicode character!
print(len(test_string))
print(len(utf8_encoded))
print(utf8_encoded.decode("utf-8"))

print('='*53 + '\n')

print('='*25 + "2.2.a" + '='*25)

test_string = "hello! こんにちは!"
utf16_encoded = test_string.encode("utf-16")
print(utf16_encoded)

# Get the byte values for the encoded string (integers from 0 to 255).
print(list(utf16_encoded))
# One byte does not necessarily correspond to one Unicode character!
print(len(test_string))
print(len(utf16_encoded))
print(utf16_encoded.decode("utf-16"))

print('='*53 + '\n')

print('='*25 + "2.2.b" + '='*25)

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
# print(decode_utf8_bytes_to_str_wrong("你好".encode("utf-8")))
# Not every Unicode characters are encoded in a single byte.

print('='*53 + '\n')