//! Research-only #898 RAM/CPU comparator. Not a product runtime.

mod r2;

use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::io::{self, Write};
use std::process::{Command, ExitCode};

const SCHEMA: &str = "newsroom.issue-898.observation-scan.v1";

pub(crate) fn current_rss_bytes() -> Value {
    let pid = std::process::id();
    match Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
    {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            match text.split_whitespace().next().and_then(|item| item.parse::<u64>().ok()) {
                Some(kib) => json!(kib.saturating_mul(1024)),
                None => json!("UNOBSERVED"),
            }
        }
        _ => json!("UNOBSERVED"),
    }
}

pub(crate) fn fail(reason: &str) -> ExitCode {
    let _ = writeln!(
        io::stdout(),
        "{}",
        json!({"status":"ERROR","reason":reason})
    );
    ExitCode::from(1)
}

fn r0() -> ExitCode {
    let payload = json!({
        "mode": "r0",
        "status": "OK",
        "rss_after_bytes": current_rss_bytes(),
        "outcome": {"imported": false, "work": "none"},
    });
    let _ = writeln!(io::stdout(), "{payload}");
    ExitCode::SUCCESS
}

pub(crate) fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn r1(args: &[String]) -> ExitCode {
    let Some(db) = arg_value(args, "--db") else {
        return fail("r1 requires --db");
    };
    let Some(cutoff) = arg_value(args, "--cutoff") else {
        return fail("r1 requires --cutoff");
    };
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = match Connection::open_with_flags(&db, flags) {
        Ok(conn) => conn,
        Err(err) => return fail(&format!("open: {err}")),
    };
    if let Err(err) = conn.pragma_update(None, "query_only", "ON") {
        return fail(&format!("query_only: {err}"));
    }
    let run_ids: Vec<String> = match conn.prepare("SELECT run_id FROM proving_runs ORDER BY rowid ASC")
    {
        Ok(mut stmt) => match stmt
            .query_map([], |row| row.get::<_, String>(0))
            .and_then(|mapped| mapped.collect::<Result<Vec<_>, _>>())
        {
            Ok(ids) => ids,
            Err(err) => return fail(&format!("runs: {err}")),
        },
        Err(err) => return fail(&format!("runs prepare: {err}")),
    };
    let mut bodies: Vec<Vec<u8>> = Vec::new();
    let mut rows: Vec<Value> = Vec::new();
    let sql = "SELECT source_id, url, fetched_at, status_code, body_digest, body, error \
               FROM proving_observations WHERE run_id=? AND fetched_at>=? \
               ORDER BY source_id, fetched_at, body_digest";
    for run_id in &run_ids {
        let mut stmt = match conn.prepare(sql) {
            Ok(stmt) => stmt,
            Err(err) => return fail(&format!("observe prepare: {err}")),
        };
        let mapped = stmt.query_map((run_id, cutoff.as_str()), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Vec<u8>>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        });
        let mapped = match mapped {
            Ok(mapped) => mapped,
            Err(err) => return fail(&format!("observe: {err}")),
        };
        for item in mapped {
            let (source_id, url, fetched_at, status_code, body_digest, body, error) = match item {
                Ok(item) => item,
                Err(err) => return fail(&format!("row: {err}")),
            };
            if status_code != 200 || body.is_empty() || error.is_some() {
                continue;
            }
            let mut hasher = Sha256::new();
            hasher.update(&body);
            let body_sha256 = format!("sha256:{:x}", hasher.finalize());
            rows.push(json!({
                "body_digest": body_digest,
                "body_len": body.len(),
                "body_sha256": body_sha256,
                "fetched_at": fetched_at,
                "run_id": run_id,
                "source_id": source_id,
                "status_code": status_code,
                "url": url,
            }));
            bodies.push(body);
        }
    }
    let manifest = json!({
        "row_count": rows.len(),
        "rows": rows,
        "schema": SCHEMA,
    });
    let canonical = canonical_json(&manifest);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let manifest_digest = format!("sha256:{:x}", hasher.finalize());
    let body_bytes: usize = bodies.iter().map(Vec::len).sum();
    let rss_held = current_rss_bytes();
    drop(bodies);
    let payload = json!({
        "mode": "r1",
        "status": "OK",
        "rss_held_bytes": rss_held,
        "rss_after_bytes": current_rss_bytes(),
        "outcome": {
            "body_bytes": body_bytes,
            "cutoff": cutoff,
            "manifest_digest": manifest_digest,
            "queue_claimed": false,
            "row_count": rows.len(),
            "schema": SCHEMA,
            "writable": false,
        },
    });
    let _ = writeln!(io::stdout(), "{payload}");
    ExitCode::SUCCESS
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = String::from("{");
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap());
                out.push(':');
                out.push_str(&canonical_json(&map[*key]));
            }
            out.push('}');
            out
        }
        Value::Array(items) => {
            let mut out = String::from("[");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&canonical_json(item));
            }
            out.push(']');
            out
        }
        other => serde_json::to_string(other).unwrap(),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("r0") => r0(),
        Some("r1") => r1(&args),
        Some("r2") => r2::run(&args),
        _ => fail("mode must be r0, r1 or r2"),
    }
}
