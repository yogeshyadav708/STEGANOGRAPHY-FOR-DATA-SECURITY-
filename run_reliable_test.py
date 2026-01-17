import sys, types, traceback

# Create lightweight stubs for heavy deps if missing
for m in ('cv2','numpy','imageio'):
    if m not in sys.modules:
        sys.modules[m] = types.ModuleType(m)

try:
    import reliable_steganography as rs
    inst = rs.ReliableSteganography()

    test_text = "Hello, Stego!"
    b = inst.text_to_binary(test_text)
    print('text_to_binary OK, length:', len(b))
    t = inst.binary_to_text(b)
    print('binary_to_text OK, text:', t)

    key = 'secret'
    enc = inst.simple_encrypt(test_text, key)
    print('simple_encrypt OK, base64 len:', len(enc))
    dec = inst.simple_decrypt(enc, key)
    print('simple_decrypt OK, text:', dec)

    print('\nAll non-image core functions executed successfully.')
except Exception as e:
    print('ERROR running tests:')
    traceback.print_exc()
    sys.exit(1)
