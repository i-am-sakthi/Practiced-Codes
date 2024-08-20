#!/usr/bin/env python
# coding: utf-8

# In[4]:


cipher = {'A': 'Z', 'B': 'Y', 'C': 'X', 'D': 'W', 'E': 'V', 'F': 'U', 'G': 'T', 'H': 'S', 'I': 'R', 'J': 'Q', 'K': 'P', 'L': 'O', 'M': 'N', 'N': 'M', 'O': 'L', 'P': 'K', 'Q': 'J', 'R': 'I', 'S': 'H', 'T': 'G', 'U': 'F', 'V': 'E', 'W': 'D', 'X': 'C', 'Y': 'B', 'Z': 'A'}
word = input("word =")
key = int(input("key=")
operation = input("operation=")
print(encrypt(word, key, operation))


def encrypt(word, key, operation):
    if not key.isalpha():
        print("Enter valid key")
        return

    if operation not in ['addition', 'subtraction']:
        print("Invalid Operation")
        return

    if not word.isupper():
        print("Word should be in capitals")
        return

    encrypted_word = ""
    for char in word:
        if char in cipher:
            cipher_value = cipher[char]
            if operation == 'addition':
                encrypted_char = chr(ord(cipher_value) + ord(key))
            else:
                encrypted_char = chr(ord(cipher_value) - ord(key))
            encrypted_word += encrypted_char
        else:
            encrypted_word += char

    return encrypted_word

def decrypt(word, key, operation):
    if not key.isalpha():
        print("Enter valid key")
        return

    if operation not in ['addition', 'subtraction']:
        print("Invalid Operation")
        return

    if not word.isupper():
        print("Word should be in capitals")
        return

    decrypted_word = ""
    for char in word:
        if char in cipher:
            cipher_value = cipher[char]
            if operation == 'addition':
                decrypted_char = chr(ord(cipher_value) - ord(key))
            else:
                decrypted_char = chr(ord(cipher_value) + ord(key))
            decrypted_word += decrypted_char
        else:
            decrypted_word += char

    return decrypted_word


# In[ ]:




