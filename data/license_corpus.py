"""Ground-truth license corpus: SPDX snippets, categories, and compatibility rules."""
from __future__ import annotations

LICENSE_CORPUS: dict[str, dict] = {
    "MIT": {
        "category": "permissive",
        "snippet": (
            'Permission is hereby granted, free of charge, to any person obtaining a copy '
            'of this software and associated documentation files (the "Software"), to deal '
            'in the Software without restriction, including without limitation the rights '
            'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell '
            'copies of the Software, and to permit persons to whom the Software is '
            'furnished to do so, subject to the following conditions:\n'
            'The above copyright notice and this permission notice shall be included in all '
            'copies or substantial portions of the Software.'
        ),
        "aliases": ["MIT License", "Expat"],
    },
    "Apache-2.0": {
        "category": "permissive",
        "snippet": (
            'Licensed under the Apache License, Version 2.0 (the "License"); '
            'you may not use this file except in compliance with the License. '
            'You may obtain a copy of the License at\n'
            '    http://www.apache.org/licenses/LICENSE-2.0\n'
            'Unless required by applicable law or agreed to in writing, software '
            'distributed under the License is distributed on an "AS IS" BASIS.'
        ),
        "aliases": ["Apache License 2.0", "ASL 2.0"],
    },
    "BSD-3-Clause": {
        "category": "permissive",
        "snippet": (
            'Redistribution and use in source and binary forms, with or without '
            'modification, are permitted provided that the following conditions are met:\n'
            '1. Redistributions of source code must retain the above copyright notice.\n'
            '2. Redistributions in binary form must reproduce the above copyright notice.\n'
            '3. Neither the name of the copyright holder nor the names of its contributors '
            'may be used to endorse or promote products derived from this software.'
        ),
        "aliases": ["BSD 3-Clause", "New BSD"],
    },
    "GPL-2.0-only": {
        "category": "copyleft_strong",
        "snippet": (
            'This program is free software; you can redistribute it and/or modify '
            'it under the terms of the GNU General Public License as published by '
            'the Free Software Foundation; version 2 of the License.\n'
            'This program is distributed in the hope that it will be useful, '
            'but WITHOUT ANY WARRANTY.'
        ),
        "aliases": ["GPL-2.0", "GPLv2", "GNU GPL v2"],
    },
    "GPL-3.0-only": {
        "category": "copyleft_strong",
        "snippet": (
            'This program is free software: you can redistribute it and/or modify '
            'it under the terms of the GNU General Public License as published by '
            'the Free Software Foundation, either version 3 of the License.\n'
            'This program is distributed in the hope that it will be useful.'
        ),
        "aliases": ["GPL-3.0", "GPLv3", "GNU GPL v3"],
    },
    "LGPL-2.1-only": {
        "category": "copyleft_weak",
        "snippet": (
            'This library is free software; you can redistribute it and/or '
            'modify it under the terms of the GNU Lesser General Public '
            'License as published by the Free Software Foundation; '
            'version 2.1 of the License.'
        ),
        "aliases": ["LGPL-2.1", "LGPLv2.1", "GNU LGPL 2.1"],
    },
    "AGPL-3.0-only": {
        "category": "copyleft_network",
        "snippet": (
            'This program is free software: you can redistribute it and/or modify '
            'it under the terms of the GNU Affero General Public License as published '
            'by the Free Software Foundation, either version 3 of the License.\n'
            'If you modify the Program, your modified version must prominently offer '
            'all users interacting with it remotely through a computer network an '
            'opportunity to receive the Corresponding Source.'
        ),
        "aliases": ["AGPL-3.0", "AGPLv3", "GNU AGPL"],
    },
    "MPL-2.0": {
        "category": "copyleft_weak",
        "snippet": (
            'This Source Code Form is subject to the terms of the Mozilla Public '
            'License, v. 2.0. If a copy of the MPL was not distributed with this '
            'file, You can obtain one at https://mozilla.org/MPL/2.0/.'
        ),
        "aliases": ["Mozilla Public License 2.0", "MPL2"],
    },
    "ISC": {
        "category": "permissive",
        "snippet": (
            'Permission to use, copy, modify, and/or distribute this software for any '
            'purpose with or without fee is hereby granted, provided that the above '
            'copyright notice and this permission notice appear in all copies.'
        ),
        "aliases": ["ISC License"],
    },
    "CC0-1.0": {
        "category": "public_domain",
        "snippet": (
            'The person who associated a work with this deed has dedicated the work to '
            'the public domain by waiving all of his or her rights to the work worldwide '
            'under copyright law, including all related and neighboring rights.'
        ),
        "aliases": ["CC0", "Creative Commons Zero"],
    },
    "Unlicense": {
        "category": "public_domain",
        "snippet": (
            'This is free and unencumbered software released into the public domain. '
            'Anyone is free to copy, modify, publish, use, compile, sell, or '
            'distribute this software, either in source code form or as a compiled '
            'binary, for any purpose, commercial or non-commercial.'
        ),
        "aliases": ["The Unlicense"],
    },
    "BUSL-1.1": {
        "category": "proprietary",
        "snippet": (
            'The Licensor hereby grants you the right to copy, modify, create derivative '
            'works, redistribute, and make non-production use of the Licensed Work. '
            'Effective on the Change Date, the License will convert to the Change License.'
        ),
        "aliases": ["Business Source License", "BSL"],
    },
    "PROPRIETARY": {
        "category": "proprietary",
        "snippet": (
            'All rights reserved. This software is proprietary and confidential. '
            'Unauthorized copying, distribution, or use of this software, via any '
            'medium, is strictly prohibited without prior written permission.'
        ),
        "aliases": ["All Rights Reserved", "Proprietary"],
    },
}

CONFLICT_RULES: dict[str, list[str]] = {
    "MIT":           ["GPL-2.0-only", "GPL-3.0-only", "AGPL-3.0-only"],
    "Apache-2.0":    ["GPL-2.0-only", "AGPL-3.0-only"],
    "BSD-3-Clause":  ["GPL-2.0-only", "GPL-3.0-only", "AGPL-3.0-only"],
    "PROPRIETARY":   ["GPL-2.0-only", "GPL-3.0-only", "AGPL-3.0-only", "LGPL-2.1-only", "MPL-2.0"],
    "BUSL-1.1":      ["GPL-2.0-only", "GPL-3.0-only", "AGPL-3.0-only"],
    "AGPL-3.0-only": ["PROPRIETARY", "BUSL-1.1"],
    "GPL-3.0-only":  ["PROPRIETARY", "BUSL-1.1", "Apache-2.0"],
}

SAAS_FORBIDDEN = ["AGPL-3.0-only"]

def get_conflict_severity(project_license: str, dep_license: str, distribution_type: str) -> str:
    if distribution_type == "saas" and dep_license in SAAS_FORBIDDEN:
        return "critical"
    conflicts = CONFLICT_RULES.get(project_license, [])
    if dep_license in conflicts:
        if dep_license in ["GPL-3.0-only", "AGPL-3.0-only"]:
            return "critical"
        if dep_license in ["GPL-2.0-only"]:
            return "high"
        return "medium"
    return "none"