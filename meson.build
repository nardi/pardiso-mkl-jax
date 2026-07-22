project(
  'pardiso-mkl-jax',
  'c', 'cpp', 'cython',
  version: '0.1.0',
  default_options: ['cpp_std=c++17', 'buildtype=release'],
)

# pure: false, since the pardiso_mkl_jax package ships a compiled extension.
python = import('python').find_installation(pure: false)

subdir('src/pardiso_mkl_jax')
