#!/bin/bash
until python scripts/structure_dm.py --from-file ABC-6K.txt --plms --parser_type constituency --conjunction --resume; do
    echo "'myscript.py' crashed with exit code $?. Restarting..." >&2
    sleep 1
done
