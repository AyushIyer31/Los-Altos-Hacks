-- NOTES:
-- table update: After creating or refreshing the view, update the timestamp in table_update, using the INSERT at the end of this file.
-- indexes: After creating the view, create indexes as well (the next sql file).

DROP MATERIALIZED VIEW IF EXISTS search_view;


CREATE OR REPLACE VIEW experiment_base_view AS (
SELECT

    -- id
    EXP.id,

    -- publication
    CASE WHEN PUB.id IS NOT NULL THEN
        jsonb_build_object(
            'id', PUB.id,
            'type', PUB.type,
            'title', PUB.title,
            'journal', PUB.journal,
            'year', PUB.year,
            'doi', PUB.doi,
            'pmid', PUB.pmid,
            'url', PUB.url,
            'authors', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                    'id', author.id,
                    'orcid', author.orcid,
                    'name', author.name
                ))
                FROM author
                WHERE author.id IN (SELECT author_id FROM publication_author WHERE publication_id = PUB.id)
            ), '[]'::jsonb)
        )
    END AS publication,

    -- dataset
    DAT.name AS dataset,

    -- annotations
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'type', ANN.type,
            'numValue', ANN.num_value,
            'strValue', ANN.str_value
        ))
        FROM experiment_annotation ANN
        WHERE ANN.experiment_id = EXP.id
    ), '[]'::jsonb) AS annotations

FROM experiment EXP
LEFT JOIN publication PUB ON EXP.publication_id = PUB.id
INNER JOIN dataset DAT ON EXP.dataset_id = DAT.id
);


CREATE OR REPLACE VIEW measurement_view AS (
SELECT
    MES.id AS "id",
    MES.type AS "type",
    MES.num_value AS "numValue",
    MES.str_value AS "strValue",

    -- datasets
    COALESCE((
        SELECT jsonb_agg(DISTINCT dataset.name)
        FROM dataset_measurement DM
        JOIN dataset ON dataset.id = DM.dataset_id
        WHERE DM.measurement_id = MES.id
    ), '[]'::jsonb) AS "datasets",

    -- references
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'type', ref.type,
            'accession', ref.accession
        ))
        FROM measurement_reference ref WHERE ref.measurement_id = MES.id
    ), '[]'::jsonb) AS "references"
FROM measurement MES
);


CREATE OR REPLACE VIEW sequence_base_view AS (
SELECT
    -- id
    SEQ.id,
    -- length
    length(SEQ.sequence) AS length,
    -- proteinLinks
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'protein', jsonb_build_object(
                'id', protein.id,
                'name', protein.name,
                'organism', protein.organism,
                'references', COALESCE((
                    SELECT jsonb_agg(jsonb_build_object(
                        'type', REF.type,
                        'accession', REF.accession,
                        'name', REF.name
                    ))
                    FROM protein_reference REF
                    WHERE REF.protein_id = protein.id
                ), '[]'::jsonb)
            ),
            'isoform', PS.isoform
        ))
        FROM protein_sequence PS
        JOIN protein ON protein.id = PS.protein_id
        WHERE PS.sequence_id = SEQ.id
    ), '[]'::jsonb) AS "proteinLinks"
FROM sequence SEQ
);


------------------ SEARCH VIEW ----------------
CREATE MATERIALIZED VIEW search_view AS (
WITH cte AS (
    SELECT
        measurement.experiment_id,
        measurement.sequence_id,
        mutant.id AS mutant_id,
        mutant.source_id,
        mutant.target_id,
        array_agg(measurement.id) AS measurement_id_array
    FROM measurement
    JOIN experiment ON measurement.experiment_id = experiment.id
    LEFT JOIN mutant ON measurement.mutant_id = mutant.id
    WHERE experiment.active IS TRUE
    GROUP BY experiment_id, sequence_id, mutant.id
)
SELECT
    cte.sequence_id,
    cte.mutant_id,
    cte.experiment_id,

    -- sequence
    CASE WHEN cte.sequence_id IS NOT NULL THEN (
        (SELECT to_jsonb(T) FROM (SELECT * FROM sequence_base_view WHERE id = cte.sequence_id) T)
        ||
        jsonb_build_object(
            'experiment', (
                (SELECT to_jsonb(T) FROM (SELECT * FROM experiment_base_view WHERE id = cte.experiment_id) T)
                ||
                jsonb_build_object(
                    'measurements', COALESCE((
                        SELECT jsonb_agg(to_jsonb(T)) FROM (SELECT * FROM measurement_view WHERE id = ANY(cte.measurement_id_array)) T
                    ), '[]'::jsonb)
                )
            ),
            'structures', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                    'id', structure.id,
                    'wwpdb', structure.wwpdb,
                    'afdb', structure.afdb,
                    'method', structure.method,
                    'resolution', structure.resolution
                ))
                FROM structure
                WHERE structure.id IN (
                    SELECT DISTINCT a.structure_id
                    FROM sequence_residue_mapping map
                    INNER JOIN residue as r ON map.residue_id = r.id
                    INNER JOIN chain as ch ON r.chain_id = ch.id
                    INNER JOIN assembly as a ON ch.assembly_id = a.id
                    WHERE map.sequence_id = cte.source_id
                )
            ), '[]'::jsonb)
        )
    ) END AS sequence,

    -- mutant
    CASE WHEN cte.mutant_id IS NOT NULL THEN (
        jsonb_build_object(
            'id', cte.mutant_id,
            'substitutions', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                    'position', sub.position,
                    'sourceAa', sub.source_aa,
                    'targetAa', sub.target_aa
                ))
                FROM substitution sub
                WHERE sub.mutant_id = cte.mutant_id
            ), '[]'::jsonb),
            'insertions', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                    'position', ins.position,
                    'aminoAcids', ins.amino_acids
                ))
                FROM insertion ins
                WHERE ins.mutant_id = cte.mutant_id
            ), '[]'::jsonb),
            'deletions', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                    'position', del.position,
                    'aminoAcids', del.amino_acids
                ))
                FROM deletion del
                WHERE del.mutant_id = cte.mutant_id
            ), '[]'::jsonb),
            'sourceSequence', (SELECT to_jsonb(T) FROM (
                SELECT id, length, "proteinLinks"
                FROM sequence_base_view
                WHERE id = cte.source_id
            ) T),
            'targetSequence', (SELECT to_jsonb(T) FROM (
                SELECT id, length, '[]'::jsonb AS "proteinLinks"
                FROM sequence_base_view
                WHERE id = cte.target_id
            ) T),
            'structures', COALESCE((
                WITH RESMAP AS (
                    SELECT
                        a.structure_id,
                        r.id, map.position, r.chain_id, r.struct_position, r.insertion_code,
                        r.amino_acid, r.asa, r.b_factor, r.secondary_structure
                    FROM sequence_residue_mapping map
                    INNER JOIN residue as r ON map.residue_id = r.id
                    INNER JOIN chain as ch ON r.chain_id = ch.id
                    INNER JOIN assembly as a ON ch.assembly_id = a.id
                    WHERE map.sequence_id = cte.source_id
                )
                SELECT jsonb_agg(jsonb_build_object(
                    'id', structure.id,
                    'wwpdb', structure.wwpdb,
                    'afdb', structure.afdb,
                    'method', structure.method,
                    'resolution', structure.resolution,
                    'residues', COALESCE((
                        SELECT jsonb_agg(jsonb_build_object(
                            'id', res.id,
                            'chainName', (SELECT "name" FROM chain WHERE chain.id = res.chain_id),
                            'structPosition', res.struct_position,
                            'seqPosition', (SELECT position FROM sequence_residue_mapping map2 WHERE map2.residue_id = res.id AND map2.sequence_id = cte.source_id),
                            'insertionCode', res.insertion_code,
                            'aminoAcid', res.amino_acid,
                            'asa', res.asa,
                            'bFactor', res.b_factor,
                            'secondaryStructure', res.secondary_structure,
                            'inTunnel', EXISTS (
                                SELECT 1
                                FROM void_residue VR
                                INNER JOIN tunnel ON tunnel.id = VR.void_id
                                WHERE VR.residue_id = res.id
                            ),
                            'inPocket', EXISTS (
                                SELECT 1
                                FROM void_residue VR
                                INNER JOIN pocket ON pocket.id = VR.void_id
                                WHERE VR.residue_id = res.id
                            )
                        ))
                        FROM RESMAP as res
                        WHERE res.structure_id = structure.id AND res.position = ANY(MUTATED_POSITION.arr)
                    ), '[]'::jsonb)
                ))
                FROM structure
                WHERE structure.id IN (SELECT DISTINCT structure_id FROM RESMAP)
            ), '[]'::jsonb),
            'features', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                    'id', sf.id,
                    'type', sf.type,
                    'position', sf.position,
                    'positionRange', CASE WHEN sf.position_range IS NOT NULL THEN jsonb_build_array(
                        CASE WHEN lower_inc(sf.position_range) THEN lower(sf.position_range) ELSE lower(sf.position_range) + 1 END,
                        CASE WHEN upper_inc(sf.position_range) THEN upper(sf.position_range) ELSE upper(sf.position_range) - 1 END
                    ) END,
                    'positionArray', sf.position_array,
                    'numValue', sf.num_value,
                    'strValue', sf.str_value
                ))
                FROM sequence_feature SF
                WHERE
                    SF.sequence_id = cte.source_id AND (
                        (SF.position IS NOT NULL AND SF.position = ANY(MUTATED_POSITION.arr)) OR
                        (SF.position_array IS NOT NULL AND SF.position_array && MUTATED_POSITION.arr) OR
                        (SF.position_range IS NOT NULL AND SF.position_range @> ANY(MUTATED_POSITION.arr))
                    )
            ), '[]'::jsonb),
            'experiment', (
                (SELECT to_jsonb(T) FROM (SELECT * FROM experiment_base_view WHERE id = cte.experiment_id) T)
                ||
                jsonb_build_object(
                    'measurements', COALESCE((
                        SELECT jsonb_agg(to_jsonb(T)) FROM (SELECT * FROM measurement_view WHERE id = ANY(cte.measurement_id_array)) T
                    ), '[]'::jsonb)
                )
            )
        )
    ) END AS mutant
FROM cte
LEFT JOIN LATERAL (
    WITH T AS (
        SELECT position FROM substitution WHERE mutant_id = cte.mutant_id
        UNION
        SELECT position FROM insertion WHERE mutant_id = cte.mutant_id
        UNION
        SELECT generate_series(position, position + length(amino_acids) - 1) AS position FROM deletion WHERE mutant_id = cte.mutant_id
    )
    SELECT
        array_agg(DISTINCT position) AS arr
    FROM T
) MUTATED_POSITION ON true
);


INSERT INTO table_update (name, updated_at)
VALUES ('search_view', NOW())
ON CONFLICT (name)
DO UPDATE SET updated_at = EXCLUDED.updated_at;