#!/bin/bash

cd /media/statespace/S/recording/output_synchronized
find . \( -name 'rgb_002000.*' -or -name 'rgb_004000.*' -or -name 'rgb_008000.*' -or -name 'rgb_010000.*' -or -name 'rgb_014000.*' -or -name 'rgb_016000.*' \) | cpio -pdm  /media/statespace/S/recording/question_images/
