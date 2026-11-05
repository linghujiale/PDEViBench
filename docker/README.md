# Container images

| Image | Contents | Used for |
|---|---|---|
| `Dockerfile.evaluator` | NumPy, SciPy, SymPy, jsonschema | Evaluation of SciPy submissions and construction of manufactured-solution references |
| `Dockerfile.dolfinx` | DOLFINx 0.9.0 with PETSc | The 72-task finite-element panel |

```bash
docker build -t pdevibench/evaluator:0.1 -f docker/Dockerfile.evaluator .
docker build -t pdevibench/dolfinx:0.9.0 -f docker/Dockerfile.dolfinx .
docker run --rm pdevibench/dolfinx:0.9.0 python /opt/pdevibench/smoke/poisson_unit_square.py
```
