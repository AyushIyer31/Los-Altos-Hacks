-- NOTES:
-- table update: After creating or refreshing the view, update the timestamp in table_update, using the INSERT at the end of this file.
-- indexes: After creating the view, create indexes as well (the next sql file).

DROP MATERIALIZED VIEW IF EXISTS search_index_view;


CREATE MATERIALIZED VIEW search_index_view AS (
WITH cte AS (
    (
        SELECT
            NULL AS sequence_id,
            source_id AS source_sequence_id,
            target_id AS target_sequence_id,
            mutant_id AS mutant_id,
            experiment_id,
            measurement_id_array
        FROM mutant
        JOIN (
            SELECT mutant_id, experiment_id, array_agg(measurement.id) AS measurement_id_array
            FROM measurement
            JOIN experiment ON measurement.experiment_id = experiment.id
            WHERE mutant_id IS NOT NULL AND experiment.active IS TRUE
            GROUP BY mutant_id, experiment_id
        ) T ON T.mutant_id = mutant.id
    ) UNION ALL (
        SELECT
            sequence_id AS sequence_id,
            sequence_id AS source_sequence_id,
            sequence_id AS target_sequence_id,
            NULL AS mutant_id,
            experiment_id as experiment_id,
            measurement_id_array
        FROM "sequence"
        JOIN (
            SELECT sequence_id, experiment_id, array_agg(measurement.id) AS measurement_id_array
            FROM measurement
            JOIN experiment ON measurement.experiment_id = experiment.id
            WHERE sequence_id IS NOT NULL AND experiment.active IS TRUE
            GROUP BY sequence_id, experiment_id
        ) T ON T.sequence_id = sequence.id
    )
)
SELECT
    -- Columns which don't represent a search node variable, start with an underscore.

    sequence_id AS _sequence_id,
    mutant_id AS _mutant_id,
    target_sequence_id AS _target_sequence_id,
    experiment_id AS _experiment_id,

    -- === NumberVariable ===
    -- SOURCE_SEQUENCE_ID
    source_sequence_id,
    -- SEQUENCE_LENGTH
    length(TGT_SEQ.sequence) AS sequence_length,
    -- IN_TUNNEL
    IN_TUNNEL.val AS in_tunnel,
    -- IN_POCKET
    IN_POCKET.val AS in_pocket,

    -- === StringVariable ===
    -- SEQUENCE: This is a special case, we don't have a field 'sequence' directly, just '_sequence_md5'.
    TGT_SEQ.sequence_md5 AS _sequence_md5,

    -- === ArrayNumberVariable ===
    -- MUTATED_POSITION
    MUTATED_POSITION.arr AS mutated_position,
    MUTATED_POSITION.avg AS mutated_position__avg,
    MUTATED_POSITION.min AS mutated_position__min,
    MUTATED_POSITION.max AS mutated_position__max,
    -- CONSERVATION
    CONSERVATION.arr AS conservation,
    CONSERVATION.avg AS conservation__avg,
    CONSERVATION.min AS conservation__min,
    CONSERVATION.max AS conservation__max,
    -- B_FACTOR
    B_FACTOR.arr AS b_factor,
    B_FACTOR.avg AS b_factor__avg,
    B_FACTOR.min AS b_factor__min,
    B_FACTOR.max AS b_factor__max,
    -- DG
    DG.arr AS dg,
    DG.avg AS dg__avg,
    DG.min AS dg__min,
    DG.max AS dg__max,
    -- DDG
    DDG.arr AS ddg,
    DDG.avg AS ddg__avg,
    DDG.min AS ddg__min,
    DDG.max AS ddg__max,
    -- DOMAINOME_DDG
    DOMAINOME_DDG.arr AS domainome_ddg,
    DOMAINOME_DDG.avg AS domainome_ddg__avg,
    DOMAINOME_DDG.min AS domainome_ddg__min,
    DOMAINOME_DDG.max AS domainome_ddg__max,
    -- DOMAINOME_FITNESS
    DOMAINOME_FITNESS.arr AS domainome_fitness,
    DOMAINOME_FITNESS.avg AS domainome_fitness__avg,
    DOMAINOME_FITNESS.min AS domainome_fitness__min,
    DOMAINOME_FITNESS.max AS domainome_fitness__max,
    -- TM
    TM.arr AS tm,
    TM.avg AS tm__avg,
    TM.min AS tm__min,
    TM.max AS tm__max,
    -- DTM
    DTM.arr AS dtm,
    DTM.avg AS dtm__avg,
    DTM.min AS dtm__min,
    DTM.max AS dtm__max,
    -- PH
    PH.arr AS ph,
    PH.avg AS ph__avg,
    PH.min AS ph__min,
    PH.max AS ph__max,

    -- === ArrayStringVariable ===
    -- SOURCE_AMINO_ACID
    SOURCE_AMINO_ACID.arr AS source_amino_acid,
    -- TARGET_AMINO_ACID
    TARGET_AMINO_ACID.arr AS target_amino_acid,
    -- ACTIVE_SITE
    ACTIVE_SITE.arr AS active_site,
    -- BINDING_SITE
    BINDING_SITE.arr AS binding_site,
    -- PROTEIN_NAME
    PROTEIN_NAME.arr AS protein_name,
    -- ORGANISM
    ORGANISM.arr AS organism,
    -- EC_NUMBER
    EC_NUMBER.arr AS ec_number,
    -- UNIPROTKB
    UNIPROTKB.arr AS uniprotkb,
    -- INTERPRO
    INTERPRO.arr AS interpro,
    -- PUBLICATION_DOI_PMID
    PUBLICATION_DOI_PMID.arr AS publication_doi_pmid,
    -- DATASET
    DATASET.arr AS dataset,
    -- MEASURE
    MEASURE.arr AS measure,
    -- WWPDB
    WWPDB.arr AS wwpdb,

    to_tsvector('simple', (
        array_to_string((
            PROTEIN_NAME.arr ||
            ORGANISM.arr ||
            EC_NUMBER.arr ||
            UNIPROTKB.arr ||
            INTERPRO.arr ||
            PUBLICATION_DOI_PMID.arr ||
            WWPDB.arr
        ), ' ')
    )) AS _fulltext
FROM cte
-- For target sequence, we can use either INNER or LEFT JOIN, as all mutants have their target sequence
INNER JOIN sequence TGT_SEQ ON TGT_SEQ.id = target_sequence_id
-- Note: Lateral joins follow. If the joined data have array_agg, it has always one row,
--   and therefore INNER JOIN should not filter out the "cte" (main) row. However, we use LEFT JOIN to be safe.
--   Rows without measurements are already filtered out in the "cte" definition.
-- PROTEIN_ID (auxiliary)
LEFT JOIN LATERAL (
    SELECT array_agg(DISTINCT protein_id) AS arr
    FROM protein_sequence
    WHERE protein_sequence.sequence_id = cte.source_sequence_id
) PROTEIN_ID ON true
-- === ArrayNumberVariable ===
-- MUTATED_POSITION
LEFT JOIN LATERAL (
    WITH T AS (
        SELECT position FROM substitution WHERE mutant_id = cte.mutant_id
        UNION
        SELECT position FROM insertion WHERE mutant_id = cte.mutant_id
        UNION
        SELECT generate_series(position, position + length(amino_acids) - 1) AS position FROM deletion WHERE mutant_id = cte.mutant_id
    )
    SELECT
        array_agg(DISTINCT position) AS arr,
        avg(position) AS avg,
        min(position) AS min,
        max(position) AS max
    FROM T
) MUTATED_POSITION ON true
-- CONSERVATION
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT num_value) AS arr,
        avg(num_value) AS avg,
        min(num_value) AS min,
        max(num_value) AS max
    FROM sequence_feature
    WHERE
        sequence_feature.sequence_id = cte.source_sequence_id AND
        sequence_feature.position = ANY(MUTATED_POSITION.arr) AND
        type = 'CONSERVATION' AND num_value IS NOT NULL
) CONSERVATION ON true
-- B_FACTOR
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT b_factor) AS arr,
        avg(b_factor) AS avg,
        min(b_factor) AS min,
        max(b_factor) AS max
    FROM sequence_residue_mapping MAP
    INNER JOIN residue ON residue.id = MAP.residue_id
    WHERE MAP.sequence_id = cte.source_sequence_id AND MAP.position = ANY(MUTATED_POSITION.arr) AND b_factor IS NOT NULL
) B_FACTOR ON true
-- DG
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT num_value) AS arr,
        avg(num_value) AS avg,
        min(num_value) AS min,
        max(num_value) AS max
    FROM measurement
    WHERE id = ANY(cte.measurement_id_array) AND type = 'DG' AND num_value IS NOT NULL
) DG ON true
-- DDG
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT num_value) AS arr,
        avg(num_value) AS avg,
        min(num_value) AS min,
        max(num_value) AS max
    FROM measurement
    WHERE id = ANY(cte.measurement_id_array) AND type = 'DDG' AND num_value IS NOT NULL
) DDG ON true
-- DOMAINOME_DDG
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT num_value) AS arr,
        avg(num_value) AS avg,
        min(num_value) AS min,
        max(num_value) AS max
    FROM measurement
    WHERE id = ANY(cte.measurement_id_array) AND type = 'DOMAINOME_DDG' AND num_value IS NOT NULL
) DOMAINOME_DDG ON true
-- DOMAINOME_FITNESS
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT num_value) AS arr,
        avg(num_value) AS avg,
        min(num_value) AS min,
        max(num_value) AS max
    FROM measurement
    WHERE id = ANY(cte.measurement_id_array) AND type = 'DOMAINOME_FITNESS' AND num_value IS NOT NULL
) DOMAINOME_FITNESS ON true
-- TM
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT num_value) AS arr,
        avg(num_value) AS avg,
        min(num_value) AS min,
        max(num_value) AS max
    FROM measurement
    WHERE id = ANY(cte.measurement_id_array) AND type = 'TM' AND num_value IS NOT NULL
) TM ON true
-- DTM
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT num_value) AS arr,
        avg(num_value) AS avg,
        min(num_value) AS min,
        max(num_value) AS max
    FROM measurement
    WHERE id = ANY(cte.measurement_id_array) AND type = 'DTM' AND num_value IS NOT NULL
) DTM ON true
-- PH
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT num_value) AS arr,
        avg(num_value) AS avg,
        min(num_value) AS min,
        max(num_value) AS max
    FROM experiment_annotation
    WHERE experiment_annotation.experiment_id = cte.experiment_id AND type = 'PH' AND num_value IS NOT NULL
) PH ON true
-- === ArrayStringVariable ===
-- SOURCE_AMINO_ACID
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(source_aa)::varchar) AS arr
    FROM substitution
    WHERE mutant_id = cte.mutant_id
) SOURCE_AMINO_ACID ON true
-- TARGET_AMINO_ACID
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(target_aa)::varchar) AS arr
    FROM substitution
    WHERE mutant_id = cte.mutant_id
) TARGET_AMINO_ACID ON true
-- ACTIVE_SITE
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(coalesce(str_value, 'active site'))::varchar) AS arr
    FROM sequence_feature
    WHERE
        sequence_feature.sequence_id = cte.source_sequence_id AND
        sequence_feature.position = ANY(MUTATED_POSITION.arr) AND
        type = 'ACTIVE_SITE'
) ACTIVE_SITE ON true
-- BINDING_SITE
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(coalesce(str_value, 'binding site'))::varchar) AS arr
    FROM sequence_feature
    WHERE
        sequence_feature.sequence_id = cte.source_sequence_id AND
        sequence_feature.position = ANY(MUTATED_POSITION.arr) AND
        type = 'BINDING_SITE'
) BINDING_SITE ON true
-- PROTEIN_NAME
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(name)::varchar) AS arr
    FROM protein
    WHERE protein.id = ANY(PROTEIN_ID.arr) AND name IS NOT NULL
) PROTEIN_NAME ON true
-- ORGANISM
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(organism)::varchar) AS arr
    FROM protein
    WHERE protein.id = ANY(PROTEIN_ID.arr) AND organism IS NOT NULL
) ORGANISM ON true
-- EC_NUMBER
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(accession)::varchar) AS arr
    FROM protein_reference
    WHERE protein_reference.protein_id = ANY(PROTEIN_ID.arr) AND type = 'EC_NUMBER'
) EC_NUMBER ON true
-- UNIPROTKB
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(accession)::varchar) AS arr
    FROM protein_reference
    WHERE protein_reference.protein_id = ANY(PROTEIN_ID.arr) AND type = 'UNIPROTKB'
) UNIPROTKB ON true
-- INTERPRO
LEFT JOIN LATERAL (
    SELECT (
        array_agg(DISTINCT lower(accession)::varchar) ||
        array_agg(DISTINCT lower(name)::varchar) FILTER (WHERE name IS NOT NULL)
    ) AS arr
    FROM protein_reference
    WHERE protein_reference.protein_id = ANY(PROTEIN_ID.arr) AND type = 'INTERPRO'
) INTERPRO ON true
-- PUBLICATION_DOI_PMID
LEFT JOIN LATERAL (
    SELECT (
        array_agg(DISTINCT lower(doi)::varchar) FILTER (WHERE doi IS NOT NULL) ||
        array_agg(DISTINCT lower(pmid)::varchar) FILTER (WHERE pmid IS NOT NULL)
    ) AS arr
    FROM publication
    WHERE id IN (
        SELECT DISTINCT publication_id
        FROM experiment
        WHERE experiment.id = cte.experiment_id
    )
) PUBLICATION_DOI_PMID ON true
-- DATASET
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(name)::varchar) AS arr
    FROM dataset
    WHERE
        id IN (
            SELECT DISTINCT dataset_id
            FROM dataset_measurement
            WHERE measurement_id = ANY(cte.measurement_id_array)
        ) OR
        id IN (
            SELECT DISTINCT dataset_id
            FROM experiment
            WHERE id = cte.experiment_id
        )
) DATASET ON true
-- MEASURE
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(str_value)::varchar) AS arr
    FROM experiment_annotation
    WHERE
        experiment_annotation.experiment_id = cte.experiment_id AND
        type = 'MEASURE' AND str_value IS NOT NULL
) MEASURE ON true
-- WWPDB
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT lower(structure.wwpdb)::varchar) AS arr
    FROM chain
    INNER JOIN assembly ON assembly.id = chain.assembly_id
    INNER JOIN structure ON structure.id = assembly.structure_id
    WHERE chain.id IN (
        SELECT DISTINCT chain_id FROM sequence_residue_mapping
        INNER JOIN residue ON residue.id = sequence_residue_mapping.residue_id
        WHERE sequence_residue_mapping.sequence_id = cte.source_sequence_id)
) WWPDB ON true
-- === BooleanVariable ===
-- MUTATED_VOID_ID (auxiliary)
LEFT JOIN LATERAL (
    SELECT
        array_agg(DISTINCT void_residue.void_id) AS arr
    FROM sequence_residue_mapping MAP
    JOIN residue ON MAP.residue_id = residue.id
    JOIN void_residue ON residue.id = void_residue.residue_id
    WHERE
        MAP.sequence_id = cte.source_sequence_id AND
        MAP.position = ANY(MUTATED_POSITION.arr)
) MUTATED_VOID_ID ON true
-- IN_TUNNEL
LEFT JOIN LATERAL (
    SELECT
        (COUNT(*) > 0) AS val
    FROM tunnel
    WHERE id = ANY(MUTATED_VOID_ID.arr)
) IN_TUNNEL ON true
-- IN_POCKET
LEFT JOIN LATERAL (
    SELECT
        (COUNT(*) > 0) AS val
    FROM pocket
    WHERE id = ANY(MUTATED_VOID_ID.arr)
) IN_POCKET ON true
); -- end of CREATE MATERIALIZED VIEW


INSERT INTO table_update (name, updated_at)
VALUES ('search_index_view', NOW())
ON CONFLICT (name)
DO UPDATE SET updated_at = EXCLUDED.updated_at;