#!/usr/bin/env python3

import sys

if sys.argv[1] == "records":
    from smartmanuscript.records import main
    main()
elif sys.argv[1] == "train":
    from smartmanuscript.train import main
    main()
