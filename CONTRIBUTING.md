# Contributing to GazeRL

Thank you for your interest in contributing to **GazeRL**! We welcome contributions, bug reports, and enhancements from the research community.

---

## 🛠️ Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/reg-from-gaze.git
   cd reg-from-gaze
   ```

2. **Create a development virtual environment**:
   ```bash
   conda create -n gazerl-dev python=3.10 -y
   conda activate gazerl-dev
   pip install -e ".[all]"
   ```

3. **Verify the test suite**:
   ```bash
   pytest -v tests/
   ```

---

## 🧪 Testing Guidelines

Before opening a pull request, please ensure:
- All 74 automated unit tests pass without errors:
  ```bash
  pytest tests/
  ```
- End-to-end smoke tests complete cleanly:
  ```bash
  python scripts/train.py --smoke-test
  python scripts/eval.py --smoke-test
  ```
- If adding a new reward function, speaker model, or listener variant, please add corresponding unit tests under `tests/`.

---

## 📜 Pull Request Guidelines

1. Create a descriptive feature branch (`git checkout -b feature/my-enhancement`).
2. Commit your changes with concise, informative commit messages.
3. Push to your fork and submit a Pull Request to `main`.
4. Ensure the GitHub Actions CI workflow passes on your PR.

---

## 📄 License

By contributing to GazeRL, you agree that your contributions will be licensed under the project's [Apache License 2.0](LICENSE).
