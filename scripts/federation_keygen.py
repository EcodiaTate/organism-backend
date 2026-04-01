#!/usr/bin/env python3
"""
EcodiaOS - Federation Key & Certificate Generator

Generates the cryptographic material needed for federation:

1. Ed25519 keypair (identity signing)
2. Self-signed X.509 certificate (mutual TLS)
3. Optional: CA certificate for multi-instance deployments

Usage:
    python scripts/federation_keygen.py --instance aurora --output config/federation/
    python scripts/federation_keygen.py --instance tide --output config/federation/ --ca config/federation/ca

This generates:
    {output}/{instance}.key         - Ed25519 private key (PEM)
    {output}/{instance}.pub         - Ed25519 public key (PEM)
    {output}/{instance}.crt         - Self-signed TLS certificate (PEM)
    {output}/{instance}_tls.key     - TLS private key (RSA 4096, PEM)

With --ca:
    {output}/ca.key                 - CA private key
    {output}/ca.crt                 - CA certificate
    {output}/{instance}.crt         - CA-signed certificate
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID


def generate_ed25519_keypair(output_dir: Path, instance_name: str) -> None:
    """Generate Ed25519 keypair for federation identity signing."""
    private_key = Ed25519PrivateKey.generate()

    # Save private key
    private_key_path = output_dir / f"{instance_name}.key"
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    print(f"  Ed25519 private key: {private_key_path}")

    # Save public key
    public_key = private_key.public_key()
    public_key_path = output_dir / f"{instance_name}.pub"
    public_key_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print(f"  Ed25519 public key:  {public_key_path}")


def generate_tls_certificate(
    output_dir: Path,
    instance_name: str,
    ca_key: rsa.RSAPrivateKey | None = None,
    ca_cert: x509.Certificate | None = None,
    hostname: str = "localhost",
    validity_days: int = 365,
) -> None:
    """Generate TLS certificate for mutual TLS federation."""
    # Generate RSA key for TLS (Ed25519 not widely supported for TLS certs)
    tls_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )

    tls_key_path = output_dir / f"{instance_name}_tls.key"
    tls_key_path.write_bytes(
        tls_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    print(f"  TLS private key:     {tls_key_path}")

    # Build certificate subject
    subject = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "EcodiaOS"),
        x509.NameAttribute(NameOID.COMMON_NAME, f"eos-{instance_name}"),
    ])

    now = datetime.now(UTC)

    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .public_key(tls_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=validity_days))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName(f"eos-{instance_name}"),
                x509.DNSName("localhost"),
            ]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=False,
        )
    )

    if ca_key and ca_cert:
        builder = builder.issuer_name(ca_cert.subject)
        cert = builder.sign(ca_key, hashes.SHA256())
    else:
        builder = builder.issuer_name(subject)  # Self-signed
        cert = builder.sign(tls_key, hashes.SHA256())

    cert_path = output_dir / f"{instance_name}.crt"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"  TLS certificate:     {cert_path}")

    # Print fingerprint
    fingerprint = cert.fingerprint(hashes.SHA256()).hex()
    print(f"  Fingerprint (SHA256): {fingerprint}")


def generate_ca(output_dir: Path, validity_days: int = 3650) -> tuple:
    """Generate a CA keypair and certificate for signing instance certs."""
    ca_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )

    ca_key_path = output_dir / "ca.key"
    ca_key_path.write_bytes(
        ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    print(f"  CA private key:      {ca_key_path}")

    subject = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "EcodiaOS Federation"),
        x509.NameAttribute(NameOID.COMMON_NAME, "EcodiaOS Federation CA"),
    ])

    now = datetime.now(UTC)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=validity_days))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    ca_cert_path = output_dir / "ca.crt"
    ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    print(f"  CA certificate:      {ca_cert_path}")

    return ca_key, ca_cert


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate federation keys and certificates for EcodiaOS instances"
    )
    parser.add_argument(
        "--instance", required=True,
        help="Instance name (e.g., 'aurora', 'tide')"
    )
    parser.add_argument(
        "--output", default="config/federation",
        help="Output directory (default: config/federation)"
    )
    parser.add_argument(
        "--ca", default=None,
        help="CA directory (if set, generates CA and signs instance cert with it)"
    )
    parser.add_argument(
        "--hostname", default="localhost",
        help="Hostname for TLS certificate SAN (default: localhost)"
    )
    parser.add_argument(
        "--validity-days", type=int, default=365,
        help="Certificate validity in days (default: 365)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating federation keys for instance '{args.instance}':")
    print(f"  Output: {output_dir.resolve()}\n")

    # Generate Ed25519 identity keypair
    print("1. Ed25519 identity keypair:")
    generate_ed25519_keypair(output_dir, args.instance)
    print()

    # Generate CA if requested
    ca_key = None
    ca_cert = None
    if args.ca:
        ca_dir = Path(args.ca)
        ca_key_path = ca_dir / "ca.key"
        ca_cert_path = ca_dir / "ca.crt"

        if ca_key_path.exists() and ca_cert_path.exists():
            print("2. Loading existing CA:")
            ca_key = serialization.load_pem_private_key(
                ca_key_path.read_bytes(), password=None
            )
            ca_cert = x509.load_pem_x509_certificate(ca_cert_path.read_bytes())
            print(f"  CA loaded from {ca_dir}")
        else:
            print("2. Generating new CA:")
            ca_dir.mkdir(parents=True, exist_ok=True)
            ca_key, ca_cert = generate_ca(ca_dir)
        print()

    # Generate TLS certificate
    signed_by = "CA" if ca_key else "self"
    print(f"3. TLS certificate ({signed_by}-signed):")
    generate_tls_certificate(
        output_dir,
        args.instance,
        ca_key=ca_key,
        ca_cert=ca_cert,
        hostname=args.hostname,
        validity_days=args.validity_days,
    )

    print("\nDone. Add to config/default.yaml:\n")
    print("  federation:")
    print("    enabled: true")
    print(f"    endpoint: \"https://{args.hostname}:8002\"")
    print(f"    tls_cert_path: \"{output_dir / f'{args.instance}.crt'}\"")
    print(f"    tls_key_path: \"{output_dir / f'{args.instance}_tls.key'}\"")
    if ca_cert:
        print(f"    ca_cert_path: \"{Path(args.ca) / 'ca.crt'}\"")
    print(f"    private_key_path: \"{output_dir / f'{args.instance}.key'}\"")
    print()


if __name__ == "__main__":
    main()
