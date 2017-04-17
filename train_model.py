#!/usr/bin/env python3

import sys

if sys.argv[1] == "records":
    from smart_manuscript.records import main
    main()
elif sys.argv[1] == "train":
    from smart_manuscript.train import main
    main()
