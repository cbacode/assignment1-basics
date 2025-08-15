test_string = "hello! こんにちは!"
utf16_encoded = test_string.encode("utf-16")
print(utf16_encoded)

# Get the byte values for the encoded string (integers from 0 to 255).
print(list(utf16_encoded))
# One byte does not necessarily correspond to one Unicode character!
print(len(test_string))
print(len(utf16_encoded))
print(utf16_encoded.decode("utf-16"))

# Answer : Utf-8 is memory efficient.