#! /bin/bash

rm ../*.html
rm ../objects.inv
rm ../searchindex.js
rm -rf ../_images
rm -rf ../_sources
rm -rf ../_static
rm -rf ../reference
rm -rf ../examples
rm -rf ../getting_started

cp -r build/html/* ../
