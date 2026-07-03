# Security Policy

## Supported Versions

Only the latest `main` branch is supported.

## Reporting a Vulnerability

Please report security issues privately via GitHub's "Report a vulnerability"
feature on this repository (Security tab), rather than opening a public issue.

Include:
- A description of the issue and its potential impact.
- Steps to reproduce, if applicable.
- Affected version/commit.

We will acknowledge reports within a reasonable timeframe and work on a fix
before any public disclosure.

## Scope notes

This project processes local video files and pretrained model weights pulled
from Hugging Face by pinned revision. It does not expose a network service by
default. Running the optional Gradio Space accepts user-uploaded video files;
treat uploaded media as untrusted input.
