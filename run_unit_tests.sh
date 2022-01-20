shopt -s extglob
pytest ./iguanas/*/tests/!(*spark*)
shopt -u extglob