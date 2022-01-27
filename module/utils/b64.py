import base64
import io

def b64_to_binary(s, isDataURL=True):
    if isDataURL: 
        s = s.split(',')[1]
    return base64.b64decode(s.encode("UTF-8"))

def binary_to_b64(s, isDataURL=True, mime="image/jpeg"):
    body = base64.b64encode(s).decode().replace("'", "")

    if isDataURL:
        body = f"data:{mime};base64,{body}"

    return body

def binary_to_BytesIO(s):
    return io.BytesIO(s)

def b64_to_BytesIO(s, isDataURL=True):
    return io.BytesIO(b64_to_binary(s, isDataURL))

def save_binary(s, filename):
    with open(filename, "wb") as f:
        f.write(s)

def save_b64(s, filename, isDataURL=True):
    save_binary(b64_to_binary(s, isDataURL), filename)
