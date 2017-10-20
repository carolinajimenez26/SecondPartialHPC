# Filtro Sobel

## Para compilar:

1. Crear una carpeta con el nombre __build__
2. Moverse a esa carpeta
3. Escribir en el terminal:

```bash
cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
make
```

Esto generará el ejecutable _sharedMem.out_

## Para ejecutar:

### Con slurn

```bash
sbatch  SharedMem.sh
```

### Local
En la carpeta _build_ escribir lo siguiente:

```bash
./global.out imagePath
```
