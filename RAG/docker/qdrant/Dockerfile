# Qdrant with ONE collection baked in: regrag_parentdoc_v1 
#
# BUILD CONTEXT IS ./qdrant_storage (set in docker-compose.build.yml), so the
# COPY below reads the live collection folder directly.
#
# WHY A FOLDER COPY AND NOT A SNAPSHOT (2026-10-05): every snapshot taken from
# the dev Qdrant (storage on a Windows bind mount) stored ALL small files -
# config.json, segment.json, wal/first-index ... - as "sparse" entries with no
# data, so they unpacked as zeros and Qdrant crashed. Observed in two separate
# snapshots. Cause of the sparse marking not proven; the folder copy avoids it.
#
# The tag must be the SAME version that wrote the folder (v1.18.3).
ARG QDRANT_TAG
FROM qdrant/qdrant:${QDRANT_TAG}
COPY collections/regrag_parentdoc_v1 /qdrant/storage/collections/regrag_parentdoc_v1
# No CMD override: the base image's ./entrypoint.sh starts Qdrant, which loads
# every collection it finds under /qdrant/storage/collections.
