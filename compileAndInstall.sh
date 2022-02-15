python setup.py sdist
rm -r pymicrodose.egg-info
mv dist/* .
rm -r dist/
pip install pymicrodose-1.1.0.tar.gz
