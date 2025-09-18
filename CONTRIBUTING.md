# Contributing to BrainTome

Thank you for your interest in contributing to BrainTome! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Git
- Basic understanding of medical imaging and deep learning

### Development Setup

1. **Fork the repository**
```bash
git clone https://github.com/yourusername/BrainTome.git
cd BrainTome
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 🛠️ Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Code Formatting
```bash
# Format code
black src/ preprocessing/

# Check style
flake8 src/ preprocessing/

# Type checking
mypy src/
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 📝 Contribution Types

### Bug Reports
- Use the bug report template
- Include system information
- Provide minimal reproducible example
- Include error messages and stack traces

### Feature Requests
- Use the feature request template
- Explain the use case and benefits
- Provide implementation suggestions if possible

### Code Contributions
- Create a new branch for your feature
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

## 🔄 Pull Request Process

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
- Write clean, documented code
- Add tests for new functionality
- Update documentation

3. **Test your changes**
```bash
python -m pytest tests/
black src/ preprocessing/
flake8 src/ preprocessing/
```

4. **Commit your changes**
```bash
git add .
git commit -m "feat: add your feature description"
```

5. **Push and create PR**
```bash
git push origin feature/your-feature-name
```

## 📋 Commit Message Guidelines

Use conventional commits format:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

## 🧪 Areas for Contribution

### High Priority
- [ ] Model architecture improvements
- [ ] Data augmentation techniques
- [ ] Performance optimization
- [ ] Documentation improvements

### Medium Priority
- [ ] Additional evaluation metrics
- [ ] Visualization enhancements
- [ ] Code refactoring
- [ ] Test coverage improvement

### Low Priority
- [ ] UI/UX improvements
- [ ] Additional dataset support
- [ ] Deployment scripts
- [ ] Docker containerization

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Medical Image Analysis Best Practices](https://link-to-resource)
- [BraTS Challenge Guidelines](https://www.med.upenn.edu/cbica/brats2024/)

## 🤝 Community

- Join our discussions in GitHub Issues
- Follow coding standards and be respectful
- Help others learn and grow

## ❓ Questions?

If you have questions about contributing, please:
1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Contact the maintainers directly

Thank you for contributing to BrainTome! 🧠✨