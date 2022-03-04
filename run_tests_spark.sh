if [ $# -eq 0 ]
  then
    echo "Notebook kernel not provided - this is required to run notebook tests."
    exit 1
fi
pytest -k "spark" iguanas --nbmake --nbmake-kernel=$1 --doctest-modules -W ignore::DeprecationWarning
pytest -k "spark" examples --nbmake --nbmake-kernel=$1 -W ignore::DeprecationWarning