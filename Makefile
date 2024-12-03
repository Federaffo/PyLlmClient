# Makefile for checking PEP8 style using flake8

.PHONY: check

check:
	@echo "Checking PEP8 style with flake8..."
	flake8 src/llmClient.py src/mock_models.py