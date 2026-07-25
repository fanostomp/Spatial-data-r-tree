# Spatial Data R-tree

Υλοποίηση R-tree για χωρικά δεδομένα σημείων με STR bulk loading και υποστήριξη για window range, distance range και k-nearest-neighbour queries.

Η πλήρης περιγραφή της εργασίας βρίσκεται στο [report.md](report.md).

## Εκτέλεση

```bash
python meros1.py Beijing_restaurants.txt rtree.csv
python meros2.py window rtree.csv windowRangeQueries.txt
python meros2.py distance rtree.csv distanceRangeQueries.txt
python meros2.py knn rtree.csv NNQueries.txt 10
```

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```
