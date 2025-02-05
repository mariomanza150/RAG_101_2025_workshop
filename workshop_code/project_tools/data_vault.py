#!/usr/bin/env python3
"""
This script encrypts and decrypts a JSON file named data.json.
The user can set their own password on the file.
Usage:
    To encrypt:   python script.py encrypt
    To decrypt:   python script.py decrypt
"""

import json
import os
import base64
import argparse
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet


def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derive a 32-byte key from the provided password and salt using PBKDF2HMAC.
    The key is then base64 urlsafe encoded (which Fernet requires).
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_json_to_file(json_obj: object, password: str, filename: str) -> None:
    """
    Encrypts a JSON-serializable object using a password and writes the
    encrypted data (prefixed with a 16-byte salt) to the specified file.
    """
    # Convert the JSON object to bytes
    data = json.dumps(json_obj, indent=4).encode('utf-8')
    # Generate a random 16-byte salt
    salt = os.urandom(16)
    # Derive a key from the password and salt
    key = derive_key(password, salt)
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data)
    # Write the salt (first 16 bytes) followed by the encrypted data to the file
    with open(filename, 'wb') as file_out:
        file_out.write(salt + encrypted_data)
    print(f"File '{filename}' encrypted successfully.")

def decrypt_json_from_file(password: str, filename: str) -> object:
    """
    Reads the encrypted file, extracts the salt, derives the key from the password,
    decrypts the data, and returns the JSON object.
    """
    with open(filename, 'rb') as file_in:
        file_data = file_in.read()
    # The first 16 bytes are the salt
    salt = file_data[:16]
    encrypted_data = file_data[16:]
    # Derive the key using the same salt
    key = derive_key(password, salt)
    fernet = Fernet(key)
    # Decrypt the data
    decrypted_data = fernet.decrypt(encrypted_data)
    json_obj = json.loads(decrypted_data.decode('utf-8'))
    return json_obj

def encrypt():
    password = input("Enter a password to encrypt the file: ")
    # Encrypt the 'questions.json' object into a file.
    with open('data.json') as file:
        data = json.load(file)
    encrypt_json_to_file(data, password, 'data.dat')

def decrypt(pswd):
    try:
        decrypted_json = decrypt_json_from_file(pswd, 'data.dat')
        print("Decrypted JSON content in file!")
        with open('data.json', 'w') as file:
            json.dump(decrypted_json, file, indent=2)
        return True
    except Exception as e:
        print("Failed to decrypt the file. Check that the password is correct and the file is valid.")
        print("Error details:", e)
        return False

def tool_run(mode='decrypt'):
    if mode == 'encrypt':
        encrypt()
    password = input("Ingresa la pswd (si aun no la tienes solo presiona enter): ")
    if "py" in password:
        flag = decrypt(password)
        if not flag:
            print(
                "¡Espera al taller! O intenta descifrarme, el código está en data_vault.py"
            )
        return flag

def main():
    parser = argparse.ArgumentParser(
        description="Encrypt or decrypt a JSON file containing questions using a user-defined password."
    )
    parser.add_argument(
        'mode', choices=['encrypt', 'decrypt'],
        help="Mode: 'encrypt' to encrypt the questions into a file, 'decrypt' to decrypt an encrypted file."
    )
    args = parser.parse_args()

    if args.mode == 'encrypt':
        encrypt()
    elif args.mode == 'decrypt':
        password = input("Enter the password to decrypt the file: ")
        decrypt(password)

if __name__ == "__main__":
    main()
