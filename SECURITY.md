# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | ✅ |
| Previous minor | ✅ |
| Older versions | ❌ |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly:

1. **Do not** create a public issue
2. Email us at **security@tern.example.com** (or use GitHub's private vulnerability reporting)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will:
- Acknowledge receipt within 48 hours
- Provide a timeline for fix
- Keep you informed of progress
- Credit you in the advisory (if desired)

## Security Considerations

### Model Loading

- Only load models from trusted sources
- Verify model checksums when possible
- The application does not execute model code - it only performs tensor operations

### API Server

- Use API keys in production
- Bind to localhost (127.0.0.1) unless external access is required
- Use a reverse proxy (nginx, Caddy) with TLS for production deployments
- Rate limiting is recommended

### Data Privacy

- All inference runs locally on your machine
- No telemetry or data collection by default
- Model downloads go directly to Hugging Face (or your configured source)
- Chat history is stored locally in plain text files

### Dependencies

We regularly audit dependencies for vulnerabilities:

```bash
cargo audit
```

## Threat Model

### In Scope

- Remote code execution via malicious model files
- Denial of service via crafted inputs
- Information disclosure through error messages
- API authentication bypass

### Out of Scope

- Physical access to the machine
- Compromised operating system
- Malicious models that produce harmful outputs (this is a model safety issue, not application security)
- Side-channel attacks (timing, power analysis)

## Secure Development Practices

- All code reviewed before merge
- Dependency scanning in CI
- Fuzzing for parsers (tokenizers, model loaders)
- Static analysis with clippy and rustsec

## Disclosure Timeline

1. Day 0: Vulnerability reported
2. Day 1-2: Acknowledgment and triage
3. Day 3-7: Root cause analysis and fix development
4. Day 7-14: Testing and preparation
5. Day 14: Coordinated disclosure with advisory
6. Day 14+: Public disclosure after users have time to update

## Contact

Security team: security@tern.example.com
PGP key: Available on request