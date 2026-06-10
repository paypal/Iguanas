
rm -rf build && python3.14 -m sphinx -M html source build

perl -pi -e 's|<img alt="Iguanas Logo" class="align-center" src="_images/IGUANAS_LOGO.png" />|<img alt="Iguanas Logo" class="align-center" src="_static/IGUANAS_LOGO.png" />|' build/html/index.html
