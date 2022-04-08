import sys

if sys.argv[1] == 'colab':
    sys.path.append('tp/src')
else:
    sys.path.append('src')

print("Hello, from init file")
