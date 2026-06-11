//! Process-wide debug / kill-switch flags, read once from the environment.
//!
//! Each flag mirrors a `LITEPARSE_*` env var. `env::var` scans the process
//! environment table and allocates on a miss, so per-line hot paths cache the
//! lookup in a `LazyLock` instead of re-reading on every call. Each static is
//! `true` when the variable is **set** (matching the historical
//! `env::var(..).is_ok()`); "disable" call sites negate it.

use std::sync::LazyLock;

macro_rules! env_set_flag {
    ($(#[$m:meta])* $name:ident, $var:literal) => {
        $(#[$m])*
        pub(super) static $name: LazyLock<bool> =
            LazyLock::new(|| std::env::var($var).is_ok());
    };
}

env_set_flag!(DEBUG_MD, "LITEPARSE_DEBUG_MD");
env_set_flag!(DEBUG_TABLE, "LITEPARSE_DEBUG_TABLE");
env_set_flag!(DEBUG_RULED, "LITEPARSE_DEBUG_RULED");
env_set_flag!(DEBUG_CROSS_REGION, "LITEPARSE_DEBUG_CROSS_REGION");
env_set_flag!(DISABLE_GLOBAL_RULED, "LITEPARSE_DISABLE_GLOBAL_RULED");
env_set_flag!(DISABLE_HEADING_GUARDS, "LITEPARSE_DISABLE_HEADING_GUARDS");
env_set_flag!(
    DISABLE_CROSS_REGION_TABLES,
    "LITEPARSE_DISABLE_CROSS_REGION_TABLES"
);
env_set_flag!(DISABLE_CLUSTER_MERGE, "LITEPARSE_DISABLE_CLUSTER_MERGE");
env_set_flag!(DISABLE_STRADDLE_GUARD, "LITEPARSE_DISABLE_STRADDLE_GUARD");
