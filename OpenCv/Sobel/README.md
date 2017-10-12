# Filtro Sobel

### Para compilar:

1. Crear una carpeta con el nombre __build__
2. Moverse a esa carpeta
3. Escribir en el terminal:

```bash
cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
make
```

Esto generará el ejecutable _a.out_

### Para ejecutar:

# Con Slurn

En la carpeta _build_:

```bash
sbatch sobel.sh ..
```

#local

En la carpeta _build_:

```bash
./Sobel.out imagePath
```
