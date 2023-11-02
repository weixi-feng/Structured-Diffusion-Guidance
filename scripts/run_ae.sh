#!/bin/bash
until python scripts/ae_dm.py --from-file $1 --plms --parser_type constituency --conjunction --resume; do
    echo "'myscript.py' crashed with exit code $?. Restarting..." >&2
    sleep 1
done
