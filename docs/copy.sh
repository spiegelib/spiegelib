#! /bin/bash

rm ../*.html
rm ../objects.inv
rm ../searchindex.js
rm -rf ../_images
rm -rf ../_sources
rm -rf ../_static

cp -r build/html/* ../
