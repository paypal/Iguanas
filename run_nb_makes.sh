shopt -s extglob
pytest ./examples/!(*spark*) --nbmake --nbmake-kernel=$1
pytest ./iguanas/*/examples/!(*spark*) --nbmake --nbmake-kernel=$1
shopt -u extglob