//! Bounded useful-output comparator. Research-only; not a product runtime.

use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::process::ExitCode;

use crate::{arg_value, current_rss_bytes, fail};

const SCHEMA: &str = "newsroom.issue-898.bounded-units.v1";
const MAX_HEADLINE_CHARS: usize = 240;
const MAX_DRAFTING_BODY_CHARS: usize = 4_000;
const MAX_EPISODE_BYTES: usize = 8 * 1024;

#[derive(Clone, Debug)]
struct SourceItem {
    source_id: String,
    item_key: String,
    headline: String,
    drafting_body: String,
    canonical_url: String,
    published_at: Option<String>,
    updated_at: Option<String>,
    corpus_body: String,
}

#[derive(Clone, Debug)]
struct UnitRow {
    chunk_count: usize,
    chunk_digest: String,
    chunk_ordinal: i64,
    ingest_id: String,
    item_key: String,
    observation_digest: String,
    observed_at: String,
    predecessor_ingest_id: Option<String>,
    proving_run_id: String,
    published_at: Option<String>,
    representation_digest: String,
    revision_digest: String,
    revision_id: String,
    source_id: String,
    updated_at: Option<String>,
}

pub(crate) fn run(args: &[String]) -> ExitCode {
    let Some(db) = arg_value(args, "--db") else {
        return fail("r2 requires --db");
    };
    let Some(spec_path) = arg_value(args, "--spec") else {
        return fail("r2 requires --spec");
    };
    let spec_text = match fs::read_to_string(&spec_path) {
        Ok(text) => text,
        Err(err) => return fail(&format!("spec read: {err}")),
    };
    let spec: Value = match serde_json::from_str(&spec_text) {
        Ok(value) => value,
        Err(err) => return fail(&format!("spec json: {err}")),
    };
    if let Some(reason) = spec_forbidden(&spec) {
        return fail(reason);
    }
    let source_id = spec_string(&spec, "source_id");
    let item_key = spec_string(&spec, "item_key");
    let revision_digest = spec_string(&spec, "revision_digest");
    let published_at = spec_string(&spec, "published_at");
    let updated_at = spec_string(&spec, "updated_at");
    let configuration_digest = spec_string(&spec, "configuration_digest");
    let temporal = spec_string(&spec, "temporal_policy_version");
    if source_id.is_empty() || configuration_digest.is_empty() || temporal.is_empty() {
        return fail("r2 spec missing source_id, configuration_digest or temporal_policy_version");
    }
    let keys = match spec.get("keys").and_then(Value::as_array) {
        Some(items) => items.clone(),
        None => return fail("r2 spec keys must be an array"),
    };
    if keys.is_empty() {
        let payload = json!({
            "mode": "r2",
            "status": "HOLD",
            "rss_after_bytes": current_rss_bytes(),
            "outcome": {
                "body_bytes": 0,
                "manifest_digest": "UNOBSERVED",
                "queue_claimed": false,
                "reason": "R2 spec has no unit_refs coordinates",
                "row_count": 0,
                "schema": SCHEMA,
                "status": "HOLD",
                "unit_count": 0,
                "writable": false,
            },
        });
        println!("{payload}");
        return ExitCode::SUCCESS;
    }

    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = match Connection::open_with_flags(&db, flags) {
        Ok(conn) => conn,
        Err(err) => return fail(&format!("open: {err}")),
    };
    if let Err(err) = conn.pragma_update(None, "query_only", "ON") {
        return fail(&format!("query_only: {err}"));
    }

    let mut collected: Vec<UnitRow> = Vec::new();
    let mut body_bytes: usize = 0;
    let mut rows_found: usize = 0;
    let sql = "SELECT source_id, url, fetched_at, status_code, body_digest, body, error \
               FROM proving_observations WHERE run_id=? AND source_id=? AND body_digest=?";
    for key in &keys {
        let run_id = key.get("run_id").and_then(Value::as_str).unwrap_or("");
        let digest = key
            .get("observation_digest")
            .and_then(Value::as_str)
            .unwrap_or("");
        if run_id.is_empty() || digest.is_empty() {
            continue;
        }
        let mut stmt = match conn.prepare(sql) {
            Ok(stmt) => stmt,
            Err(err) => return fail(&format!("observe prepare: {err}")),
        };
        let mapped = stmt.query_map((run_id, source_id.as_str(), digest), |row| {
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
            let (row_source, url, fetched_at, status_code, body_digest, body, error) = match item {
                Ok(item) => item,
                Err(err) => return fail(&format!("row: {err}")),
            };
            if status_code != 200 || body.is_empty() || error.is_some() {
                continue;
            }
            body_bytes += body.len();
            rows_found += 1;
            for parsed in parse_observation(&row_source, &url, &body) {
                for unit in units_from_item(
                    &parsed,
                    &body_digest,
                    run_id,
                    &fetched_at,
                    &configuration_digest,
                    &temporal,
                ) {
                    let pub_at = unit.published_at.clone().unwrap_or_default();
                    let upd_at = unit.updated_at.clone().unwrap_or_default();
                    if unit.item_key == item_key
                        && unit.revision_digest == revision_digest
                        && pub_at == published_at
                        && upd_at == updated_at
                    {
                        collected.push(unit);
                    }
                }
            }
        }
    }
    let units = unique_units(collected);
    let comparable: Vec<Value> = units.iter().map(unit_json).collect();
    let manifest = json!({
        "row_count": rows_found,
        "schema": SCHEMA,
        "unit_count": comparable.len(),
        "units": comparable,
    });
    let digest = sha256_hex(authority_canonical(&manifest).as_bytes());
    let status = if comparable.is_empty() { "HOLD" } else { "OK" };
    let reason = if rows_found == 0 {
        Some("unit_refs coordinates did not match proving_observations rows")
    } else if comparable.is_empty() {
        Some("bounded rows were read, but parser/identity produced no matching units")
    } else {
        None
    };
    let rss_held = current_rss_bytes();
    let payload = json!({
        "mode": "r2",
        "status": status,
        "rss_held_bytes": rss_held,
        "rss_after_bytes": current_rss_bytes(),
        "outcome": {
            "body_bytes": body_bytes,
            "manifest_digest": digest,
            "queue_claimed": false,
            "reason": reason,
            "row_count": rows_found,
            "schema": SCHEMA,
            "status": status,
            "unit_count": comparable.len(),
            "writable": false,
        },
    });
    println!("{payload}");
    ExitCode::SUCCESS
}

fn spec_forbidden(spec: &Value) -> Option<&'static str> {
    let obj = spec.as_object()?;
    for key in [
        "authority_record_ids",
        "chunk_digest",
        "ingest_id",
        "predecessor_ingest_id",
        "representation_digest",
        "revision_id",
        "unit_refs",
    ] {
        if obj.contains_key(key) {
            return Some("r2 spec must not carry unit_ref oracle fields");
        }
    }
    None
}

fn spec_string(spec: &Value, key: &str) -> String {
    spec.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn parse_observation(source_id: &str, url: &str, body: &[u8]) -> Vec<SourceItem> {
    if body.is_empty() {
        return Vec::new();
    }
    let stripped = trim_start(body);
    if stripped.starts_with(b"{") || stripped.starts_with(b"[") {
        return from_json(source_id, body);
    }
    let url_l = url.to_ascii_lowercase();
    if stripped.starts_with(b"<")
        || url_l.contains("rss")
        || url_l.contains("atom")
        || url_l.ends_with(".xml")
    {
        return from_xml(source_id, body);
    }
    from_json(source_id, body)
}

fn trim_start(body: &[u8]) -> &[u8] {
    let mut index = 0;
    while index < body.len() && body[index].is_ascii_whitespace() {
        index += 1;
    }
    &body[index..]
}

fn from_json(source_id: &str, body: &[u8]) -> Vec<SourceItem> {
    let text = match std::str::from_utf8(body) {
        Ok(text) => text,
        Err(_) => return Vec::new(),
    };
    let value: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };
    if let Some(list) = value.as_array() {
        return list
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                entry.as_object().and_then(|obj| {
                    from_mapping(source_id, obj, &index.to_string())
                })
            })
            .collect();
    }
    let Some(obj) = value.as_object() else {
        return Vec::new();
    };
    if obj.contains_key("title") || obj.contains_key("base_path") {
        return from_mapping(source_id, obj, source_id)
            .into_iter()
            .collect();
    }
    obj.iter()
        .filter_map(|(key, entry)| {
            entry
                .as_object()
                .and_then(|inner| from_mapping(source_id, inner, key))
        })
        .collect()
}

fn from_mapping(
    source_id: &str,
    payload: &Map<String, Value>,
    fallback_key: &str,
) -> Option<SourceItem> {
    let title = payload
        .get("title")
        .or_else(|| payload.get("name"))
        .or_else(|| payload.get("code"))
        .and_then(Value::as_str)?;
    let headline = plain(title);
    if headline.is_empty() {
        return None;
    }
    let description = payload
        .get("description")
        .or_else(|| payload.get("summary"))
        .and_then(Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .unwrap_or(title);
    let retained = payload
        .get("details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("body"))
        .and_then(Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .unwrap_or(description);
    let mut url = payload
        .get("base_path")
        .or_else(|| payload.get("url"))
        .or_else(|| payload.get("link"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if url.starts_with('/') {
        url = format!("https://www.gov.uk{url}");
    }
    let key = payload
        .get("content_id")
        .or_else(|| payload.get("code"))
        .and_then(Value::as_str)
        .unwrap_or(fallback_key)
        .to_string();
    let published_raw = payload
        .get("first_published_at")
        .or_else(|| payload.get("published_at"))
        .or_else(|| payload.get("public_timestamp"))
        .and_then(Value::as_str);
    let updated_raw = payload
        .get("public_updated_at")
        .or_else(|| payload.get("updated_at"))
        .and_then(Value::as_str);
    Some(SourceItem {
        source_id: source_id.to_string(),
        item_key: key,
        headline: clip(&headline, MAX_HEADLINE_CHARS),
        drafting_body: clip(&plain(description), MAX_DRAFTING_BODY_CHARS),
        canonical_url: url,
        published_at: published_raw.and_then(parse_source_time),
        updated_at: updated_raw.and_then(parse_source_time),
        corpus_body: plain(retained),
    })
}

fn from_xml(source_id: &str, body: &[u8]) -> Vec<SourceItem> {
    let mut reader = quick_xml::Reader::from_reader(body);
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();
    let mut items = Vec::new();
    let mut in_item = false;
    let mut item_depth: usize = 0;
    let mut children: Vec<(String, String, Option<String>)> = Vec::new();
    let mut child_name: Option<String> = None;
    let mut child_href: Option<String> = None;
    let mut child_text = String::new();
    let mut child_depth: usize = 0;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => return Vec::new(),
            Ok(quick_xml::events::Event::Start(e)) => {
                let name = local_name(e.name().as_ref());
                if !in_item && (name == "item" || name == "entry") {
                    in_item = true;
                    item_depth = 1;
                    children.clear();
                    continue;
                }
                if in_item {
                    item_depth += 1;
                    if child_name.is_none() {
                        child_name = Some(name);
                        child_href = attr(&e, b"href");
                        child_text.clear();
                        child_depth = 1;
                    } else {
                        child_depth += 1;
                    }
                }
            }
            Ok(quick_xml::events::Event::Empty(e)) => {
                let name = local_name(e.name().as_ref());
                if !in_item && (name == "item" || name == "entry") {
                    continue;
                }
                if in_item && child_name.is_none() {
                    children.push((name, String::new(), attr(&e, b"href")));
                }
            }
            Ok(quick_xml::events::Event::Text(t)) => {
                if child_name.is_some() {
                    child_text.push_str(&t.unescape().unwrap_or_default());
                }
            }
            Ok(quick_xml::events::Event::CData(t)) => {
                if child_name.is_some() {
                    child_text.push_str(&String::from_utf8_lossy(t.as_ref()));
                }
            }
            Ok(quick_xml::events::Event::End(e)) => {
                let name = local_name(e.name().as_ref());
                if child_name.as_deref() == Some(name.as_str()) && child_depth == 1 {
                    children.push((
                        child_name.take().unwrap(),
                        std::mem::take(&mut child_text),
                        child_href.take(),
                    ));
                    child_depth = 0;
                } else if child_name.is_some() {
                    child_depth = child_depth.saturating_sub(1);
                }
                if in_item {
                    item_depth = item_depth.saturating_sub(1);
                    if item_depth == 0 && (name == "item" || name == "entry") {
                        if let Some(item) = item_from_children(source_id, &children) {
                            items.push(item);
                        }
                        in_item = false;
                        children.clear();
                    }
                }
            }
            _ => {}
        }
        buf.clear();
    }
    items
}

fn attr(e: &quick_xml::events::BytesStart<'_>, name: &[u8]) -> Option<String> {
    e.try_get_attribute(name)
        .ok()
        .flatten()
        .and_then(|item| String::from_utf8(item.value.into_owned()).ok())
}

fn local_name(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);
    text.rsplit('}').next().unwrap_or(&text).to_ascii_lowercase()
}

fn item_from_children(
    source_id: &str,
    children: &[(String, String, Option<String>)],
) -> Option<SourceItem> {
    let title = child_text(children, &["title"]);
    let fallback: String = children
        .iter()
        .map(|item| item.1.as_str())
        .collect::<Vec<_>>()
        .join("");
    let headline_src = if title.is_empty() {
        plain(&fallback.chars().take(240).collect::<String>())
    } else {
        title
    };
    let headline = clip(&headline_src, MAX_HEADLINE_CHARS);
    if headline.is_empty() {
        return None;
    }
    let mut link = child_text(children, &["link"]);
    if link.is_empty() {
        if let Some((_, _, href)) = children.iter().find(|item| item.0 == "link") {
            link = href.clone().unwrap_or_default();
        }
    }
    let key_raw = child_text(children, &["guid", "id"]);
    let key = if !key_raw.is_empty() {
        key_raw
    } else if !link.is_empty() {
        link.clone()
    } else {
        headline.clone()
    };
    let summary = child_text(children, &["description", "summary"]);
    let full_content = child_text(children, &["content", "encoded"]);
    let corpus_body = if !full_content.is_empty() {
        full_content
    } else if !summary.is_empty() {
        summary.clone()
    } else {
        headline.clone()
    };
    let drafting = if !summary.is_empty() {
        summary
    } else {
        corpus_body.clone()
    };
    Some(SourceItem {
        source_id: source_id.to_string(),
        item_key: key,
        headline,
        drafting_body: clip(&drafting, MAX_DRAFTING_BODY_CHARS),
        canonical_url: link,
        published_at: parse_source_time(&child_text(children, &["published", "pubdate", "date"])),
        updated_at: parse_source_time(&child_text(children, &["updated"])),
        corpus_body,
    })
}

fn child_text(children: &[(String, String, Option<String>)], names: &[&str]) -> String {
    for name in names {
        for (child, text, href) in children {
            if child != name {
                continue;
            }
            if *name == "link" {
                if let Some(href) = href {
                    if !href.is_empty() {
                        return href.clone();
                    }
                }
            }
            return plain(text);
        }
    }
    String::new()
}

fn plain(text: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in text.chars() {
        if ch == '<' {
            in_tag = true;
            out.push(' ');
            continue;
        }
        if ch == '>' {
            in_tag = false;
            continue;
        }
        if !in_tag {
            out.push(ch);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clip(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= limit {
        return trimmed.to_string();
    }
    let mut cut: String = trimmed.chars().take(limit.saturating_sub(1)).collect();
    while cut.ends_with(char::is_whitespace) {
        cut.pop();
    }
    cut.push('…');
    cut
}

fn parse_source_time(raw: &str) -> Option<String> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }
    if text.len() == 10 && text.as_bytes().get(4) == Some(&b'-') && text.as_bytes().get(7) == Some(&b'-')
    {
        return Some(format!("{text}T00:00:00.000000Z"));
    }
    parse_iso(text)
}

fn parse_iso(text: &str) -> Option<String> {
    let mut normalised = text.trim().to_string();
    if normalised.ends_with('Z') {
        normalised.pop();
    } else if let Some(stripped) = normalised.strip_suffix("+00:00") {
        normalised = stripped.to_string();
    }
    let (date, rest) = normalised.split_once('T')?;
    if date.len() != 10 {
        return None;
    }
    let (hms, frac) = match rest.split_once('.') {
        Some((hms, frac)) => (hms, frac),
        None => (rest, "000000"),
    };
    if hms.len() != 8 {
        return None;
    }
    let mut micros: String = frac.chars().filter(|ch| ch.is_ascii_digit()).collect();
    micros.truncate(6);
    while micros.len() < 6 {
        micros.push('0');
    }
    Some(format!("{date}T{hms}.{micros}Z"))
}

fn units_from_item(
    item: &SourceItem,
    observation_digest: &str,
    proving_run_id: &str,
    observed_at: &str,
    configuration_digest: &str,
    temporal: &str,
) -> Vec<UnitRow> {
    let body = if item.corpus_body.is_empty() {
        item.drafting_body.as_str()
    } else {
        item.corpus_body.as_str()
    };
    let revision_digest = content_digest(&item.headline, body, &item.canonical_url);
    let representation_digest = representation_digest_for(
        &item.source_id,
        &item.item_key,
        &revision_digest,
        item.published_at.as_deref(),
        item.updated_at.as_deref(),
    );
    let revision_id = source_revision_id(
        &item.source_id,
        &item.item_key,
        &revision_digest,
        item.published_at.as_deref(),
        item.updated_at.as_deref(),
    );
    let mut parts = Vec::new();
    if !item.headline.trim().is_empty() {
        parts.push(item.headline.trim().to_string());
    }
    if !body.trim().is_empty() {
        parts.push(body.trim().to_string());
    }
    if !item.canonical_url.trim().is_empty() {
        parts.push(item.canonical_url.trim().to_string());
    }
    let full_text = parts.join("\n");
    let chunks = chunk_text(&full_text);
    let mut predecessor: Option<String> = None;
    let mut rows = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let ordinal = (index + 1) as i64;
        let chunk_digest = content_digest("", chunk, "");
        let ingest_id = ingest_key(
            &item.source_id,
            &item.item_key,
            &revision_digest,
            &revision_id,
            &representation_digest,
            item.published_at.as_deref(),
            item.updated_at.as_deref(),
            ordinal,
            configuration_digest,
            temporal,
        );
        rows.push(UnitRow {
            chunk_count: chunks.len(),
            chunk_digest,
            chunk_ordinal: ordinal,
            ingest_id: ingest_id.clone(),
            item_key: item.item_key.clone(),
            observation_digest: observation_digest.to_string(),
            observed_at: observed_at.to_string(),
            predecessor_ingest_id: predecessor.clone(),
            proving_run_id: proving_run_id.to_string(),
            published_at: item.published_at.clone(),
            representation_digest: representation_digest.clone(),
            revision_digest: revision_digest.clone(),
            revision_id: revision_id.clone(),
            source_id: item.source_id.clone(),
            updated_at: item.updated_at.clone(),
        });
        predecessor = Some(ingest_id);
    }
    rows
}

fn unique_units(units: Vec<UnitRow>) -> Vec<UnitRow> {
    let mut selected: BTreeMap<(String, String, String, String, String, i64), UnitRow> =
        BTreeMap::new();
    for unit in units {
        let key = (
            unit.source_id.clone(),
            unit.item_key.clone(),
            unit.revision_digest.clone(),
            unit.published_at.clone().unwrap_or_default(),
            unit.updated_at.clone().unwrap_or_default(),
            unit.chunk_ordinal,
        );
        match selected.get(&key) {
            Some(previous) if previous.observed_at <= unit.observed_at => {}
            _ => {
                selected.insert(key, unit);
            }
        }
    }
    let mut out: Vec<UnitRow> = selected.into_values().collect();
    out.sort_by(|left, right| {
        (
            left.observed_at.as_str(),
            left.revision_id.as_str(),
            left.chunk_ordinal,
        )
            .cmp(&(
                right.observed_at.as_str(),
                right.revision_id.as_str(),
                right.chunk_ordinal,
            ))
    });
    out
}

fn unit_json(unit: &UnitRow) -> Value {
    json!({
        "chunk_count": unit.chunk_count,
        "chunk_digest": unit.chunk_digest,
        "chunk_ordinal": unit.chunk_ordinal,
        "ingest_id": unit.ingest_id,
        "item_key": unit.item_key,
        "observation_digest": unit.observation_digest,
        "predecessor_ingest_id": unit.predecessor_ingest_id,
        "proving_run_id": unit.proving_run_id,
        "published_at": unit.published_at,
        "representation_digest": unit.representation_digest,
        "revision_digest": unit.revision_digest,
        "revision_id": unit.revision_id,
        "source_id": unit.source_id,
        "updated_at": unit.updated_at,
    })
}

fn content_digest(headline: &str, body: &str, canonical_url: &str) -> String {
    sha256_hex(
        authority_canonical(&json!({
            "body": body,
            "canonical_url": canonical_url,
            "headline": headline,
        }))
        .as_bytes(),
    )
}

fn representation_digest_for(
    source_id: &str,
    item_key: &str,
    revision_digest: &str,
    published_at: Option<&str>,
    updated_at: Option<&str>,
) -> String {
    sha256_hex(
        authority_canonical(&json!({
            "item_key": item_key,
            "published_at": published_at,
            "revision_digest": revision_digest,
            "source_id": source_id,
            "updated_at": updated_at,
        }))
        .as_bytes(),
    )
}

fn source_revision_id(
    source_id: &str,
    item_key: &str,
    revision_digest: &str,
    published_at: Option<&str>,
    updated_at: Option<&str>,
) -> String {
    typed_id(&[
        json!("revision"),
        json!(source_id),
        json!(item_key),
        json!(revision_digest),
        json!(published_at.unwrap_or("")),
        json!(updated_at.unwrap_or("")),
    ])
}

fn ingest_key(
    source_id: &str,
    item_key: &str,
    content_digest_value: &str,
    revision_id: &str,
    representation_digest: &str,
    published_at: Option<&str>,
    updated_at: Option<&str>,
    chunk_ordinal: i64,
    configuration_digest: &str,
    temporal: &str,
) -> String {
    let payload = json!({
        "chunk_ordinal": chunk_ordinal,
        "configuration": configuration_digest,
        "content_digest": content_digest_value,
        "item_key": item_key,
        "published_at": published_at,
        "representation_digest": representation_digest,
        "revision_id": revision_id,
        "source_id": source_id,
        "temporal": temporal,
        "updated_at": updated_at,
    });
    uuid4_from_sha256(&sha256_hex(authority_canonical(&payload).as_bytes()))
}

fn typed_id(parts: &[Value]) -> String {
    uuid4_from_sha256(&sha256_hex(
        authority_canonical(&Value::Array(parts.to_vec())).as_bytes(),
    ))
}

fn chunk_text(text: &str) -> Vec<String> {
    let data = text.as_bytes();
    if data.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let mut end = (offset + MAX_EPISODE_BYTES).min(data.len());
        while end > offset && std::str::from_utf8(&data[offset..end]).is_err() {
            end -= 1;
        }
        if end == offset {
            break;
        }
        chunks.push(String::from_utf8(data[offset..end].to_vec()).unwrap());
        offset = end;
    }
    chunks
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn uuid4_from_sha256(digest: &str) -> String {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    let raw = hex.as_bytes();
    let mut bytes = [0u8; 16];
    for index in 0..16 {
        let start = index * 2;
        bytes[index] = u8::from_str_radix(std::str::from_utf8(&raw[start..start + 2]).unwrap(), 16)
            .unwrap();
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

fn authority_canonical(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => emit_json_string(text),
        Value::Array(items) => {
            let mut out = String::from("[");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&authority_canonical(item));
            }
            out.push(']');
            out
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = String::from("{");
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&emit_json_string(key));
                out.push(':');
                out.push_str(&authority_canonical(&map[*key]));
            }
            out.push('}');
            out
        }
    }
}

fn emit_json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
