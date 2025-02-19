from lib.hello import Greeting

# Create an instance of the Greeting class
test = Greeting()
print(test)
# Output: Greeting(message='')

# Set the 'message' field
test.message = "Hey!"
print(test)
# Output: Greeting(message="Hey!")

# Serialize the instance to bytes
serialized = bytes(test)
print(serialized)
# Output: b'\n\x04Hey!'