SET @schema_name = DATABASE();
SET @push_rules_table = IF(
    (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = @schema_name AND table_name = '_reclive_push_rules_cutover') = 1,
    '_reclive_push_rules_cutover',
    'push_rules'
);

SET @statement = CONCAT('CREATE TABLE IF NOT EXISTS `', @push_rules_table, '` (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    endpoint_hash BINARY(32) NULL,
    subscription_json JSON NOT NULL,
    facility_id INT NOT NULL,
    section_key VARCHAR(80) NOT NULL,
    threshold TINYINT UNSIGNED NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    expires_at DATETIME(6) NULL,
    status VARCHAR(32) NULL,
    active_identity TINYINT GENERATED ALWAYS AS (CASE WHEN status IN (''pending'', ''claimed'') THEN 1 ELSE NULL END) STORED,
    claimed_at DATETIME(6) NULL,
    sent_at DATETIME(6) NULL,
    finalized_at DATETIME(6) NULL,
    failure_code VARCHAR(64) NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;

SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = @push_rules_table AND column_name = 'endpoint_hash') = 0, CONCAT('ALTER TABLE `', @push_rules_table, '` ADD COLUMN endpoint_hash BINARY(32) NULL'), 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = @push_rules_table AND column_name = 'expires_at') = 0, CONCAT('ALTER TABLE `', @push_rules_table, '` ADD COLUMN expires_at DATETIME(6) NULL'), 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = @push_rules_table AND column_name = 'status') = 0, CONCAT('ALTER TABLE `', @push_rules_table, '` ADD COLUMN status VARCHAR(32) NULL'), 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = @push_rules_table AND column_name = 'active_identity') = 0, CONCAT('ALTER TABLE `', @push_rules_table, '` ADD COLUMN active_identity TINYINT GENERATED ALWAYS AS (CASE WHEN status IN (''pending'', ''claimed'') THEN 1 ELSE NULL END) STORED'), 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = @push_rules_table AND column_name = 'claimed_at') = 0, CONCAT('ALTER TABLE `', @push_rules_table, '` ADD COLUMN claimed_at DATETIME(6) NULL'), 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = @push_rules_table AND column_name = 'sent_at') = 0, CONCAT('ALTER TABLE `', @push_rules_table, '` ADD COLUMN sent_at DATETIME(6) NULL'), 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = @push_rules_table AND column_name = 'finalized_at') = 0, CONCAT('ALTER TABLE `', @push_rules_table, '` ADD COLUMN finalized_at DATETIME(6) NULL'), 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
SET @statement = IF((SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = @schema_name AND table_name = @push_rules_table AND column_name = 'failure_code') = 0, CONCAT('ALTER TABLE `', @push_rules_table, '` ADD COLUMN failure_code VARCHAR(64) NULL'), 'SELECT 1');
PREPARE migration_statement FROM @statement;
EXECUTE migration_statement;
DEALLOCATE PREPARE migration_statement;
