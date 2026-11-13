// Copyright IBM Corp. 2024, 2026
// SPDX-License-Identifier: BUSL-1.1

package resolver

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/singleflight"

	"github.com/hashicorp/consul/acl"
)

// DefaultNamespace is used when a token or policy set does not name a
// namespace explicitly. Namespaces are a hard boundary for this resolver:
// secrets, policy names and cache entries never leak across namespaces.
const DefaultNamespace = "default"

// LocalToken is a token record held by the in-memory LocalStore.
type LocalToken struct {
	AccessorID string
	SecretID   string
	Namespace  string
	// Policies names policies within the token's own namespace.
	Policies []string
	// ExpiresAt is the absolute expiry time; the zero value never expires.
	ExpiresAt time.Time
}

func (t *LocalToken) expired(now time.Time) bool {
	return !t.ExpiresAt.IsZero() && !now.Before(t.ExpiresAt)
}

func normalizeNamespace(ns string) string {
	if ns == "" {
		return DefaultNamespace
	}
	return ns
}

// localSnapshot is an immutable view of the whole policy+token state.
// Readers always load a snapshot atomically, so a resolution can never
// observe half of a policy set: every rule, token and version it sees
// comes from exactly one generation.
type localSnapshot struct {
	version uint64
	// policies maps namespace -> policy name -> parsed policy.
	policies map[string]map[string]*acl.Policy
	// tokens maps namespace+"\x00"+accessorID -> token.
	tokens map[string]*LocalToken
	// secrets maps namespace+"\x00"+secretID -> accessorID.
	secrets map[string]string
}

func nsKey(ns, id string) string { return ns + "\x00" + id }

// LocalStore is an in-memory, versioned store of policy text and tokens.
//
// Every mutation is copy-on-write: a brand new immutable snapshot is built
// and swapped in atomically, and the version increases monotonically.
// Policy text is fully parsed and validated before the swap, so a failed
// load leaves the previous generation untouched. One generation of history
// is retained to support Rollback.
type LocalStore struct {
	mu      sync.Mutex // serializes writers; readers never lock
	counter atomic.Uint64
	cur     atomic.Pointer[localSnapshot]
	prev    atomic.Pointer[localSnapshot]
}

func NewLocalStore() *LocalStore {
	s := &LocalStore{}
	s.cur.Store(&localSnapshot{
		version:  s.counter.Add(1),
		policies: map[string]map[string]*acl.Policy{},
		tokens:   map[string]*LocalToken{},
		secrets:  map[string]string{},
	})
	return s
}

// Version returns the current generation.
func (s *LocalStore) Version() uint64 { return s.cur.Load().version }

// Versions returns the current and previous (rollback) generations so that
// caches can evict entries belonging to any other generation.
func (s *LocalStore) Versions() (cur, prev uint64) {
	cur = s.cur.Load().version
	if p := s.prev.Load(); p != nil {
		prev = p.version
	}
	return cur, prev
}

func (s *LocalStore) snapshot() *localSnapshot { return s.cur.Load() }

// clone returns a mutable copy of the snapshot's maps under a new version.
func (snap *localSnapshot) clone(version uint64) *localSnapshot {
	out := &localSnapshot{
		version:  version,
		policies: make(map[string]map[string]*acl.Policy, len(snap.policies)),
		tokens:   make(map[string]*LocalToken, len(snap.tokens)),
		secrets:  make(map[string]string, len(snap.secrets)),
	}
	for ns, set := range snap.policies {
		cpy := make(map[string]*acl.Policy, len(set))
		for name, p := range set {
			cpy[name] = p
		}
		out.policies[ns] = cpy
	}
	for k, v := range snap.tokens {
		out.tokens[k] = v
	}
	for k, v := range snap.secrets {
		out.secrets[k] = v
	}
	return out
}

// swap installs next as the current generation, keeping the old one for
// rollback. Callers must hold s.mu.
func (s *LocalStore) swap(next *localSnapshot) {
	s.prev.Store(s.cur.Load())
	s.cur.Store(next)
}

// LoadPolicies parses and validates every policy text for the namespace and
// only then atomically replaces that namespace's policy set. On any parse
// error the store is left completely unchanged, so readers never observe a
// partially applied update. Returns the new generation.
func (s *LocalStore) LoadPolicies(namespace string, texts map[string]string, conf *acl.Config) (uint64, error) {
	namespace = normalizeNamespace(namespace)

	// Parse everything up-front: an all-or-nothing boundary.
	parsed := make(map[string]*acl.Policy, len(texts))
	for name, text := range texts {
		if name == "" {
			return 0, fmt.Errorf("policy name must not be empty")
		}
		p, err := acl.NewPolicyFromSource(text, conf, nil)
		if err != nil {
			return 0, fmt.Errorf("failed to parse policy %q in namespace %q: %w", name, namespace, err)
		}
		parsed[name] = p
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	next := s.cur.Load().clone(s.counter.Add(1))
	next.policies[namespace] = parsed
	s.swap(next)
	return next.version, nil
}

// UpsertToken inserts or replaces a token. Returns the new generation.
func (s *LocalStore) UpsertToken(tok LocalToken) (uint64, error) {
	if tok.AccessorID == "" {
		return 0, fmt.Errorf("token accessor ID must not be empty")
	}
	if tok.SecretID == "" {
		return 0, fmt.Errorf("token secret ID must not be empty")
	}
	tok.Namespace = normalizeNamespace(tok.Namespace)

	s.mu.Lock()
	defer s.mu.Unlock()

	cur := s.cur.Load()
	// A secret may only ever identify one token within a namespace; reject
	// secret reuse across accessors so identities cannot be confused.
	if existing, ok := cur.secrets[nsKey(tok.Namespace, tok.SecretID)]; ok && existing != tok.AccessorID {
		return 0, fmt.Errorf("secret already bound to a different accessor in namespace %q", tok.Namespace)
	}

	next := cur.clone(s.counter.Add(1))
	// If the accessor previously existed under another secret, drop the old
	// secret mapping so the stale secret stops resolving immediately.
	if old, ok := cur.tokens[nsKey(tok.Namespace, tok.AccessorID)]; ok && old.SecretID != tok.SecretID {
		delete(next.secrets, nsKey(tok.Namespace, old.SecretID))
	}
	tok.Policies = append([]string(nil), tok.Policies...)
	next.tokens[nsKey(tok.Namespace, tok.AccessorID)] = &tok
	next.secrets[nsKey(tok.Namespace, tok.SecretID)] = tok.AccessorID
	s.swap(next)
	return next.version, nil
}

// RevokeToken removes a token by accessor ID. The token (and its secret)
// stops resolving as of the new generation. Returns the new generation and
// whether the token existed.
func (s *LocalStore) RevokeToken(namespace, accessorID string) (uint64, bool) {
	namespace = normalizeNamespace(namespace)

	s.mu.Lock()
	defer s.mu.Unlock()

	cur := s.cur.Load()
	tok, ok := cur.tokens[nsKey(namespace, accessorID)]
	if !ok {
		return cur.version, false
	}
	next := cur.clone(s.counter.Add(1))
	delete(next.tokens, nsKey(namespace, accessorID))
	delete(next.secrets, nsKey(namespace, tok.SecretID))
	s.swap(next)
	return next.version, true
}

// Rollback restores the previous generation. The restored content keeps its
// original version, which is safe because generations are immutable: a
// version number always denotes exactly one policy/token state. Returns the
// restored generation and whether a previous generation existed.
func (s *LocalStore) Rollback() (uint64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	prev := s.prev.Load()
	if prev == nil {
		return s.cur.Load().version, false
	}
	// Swap current and previous: a second rollback is a redo.
	s.prev.Store(s.cur.Load())
	s.cur.Store(prev)
	return prev.version, true
}

// AuditEvent describes one token resolution. It deliberately contains no
// token secret: the only token-derived field is a truncated SHA-256
// fingerprint, which is enough to correlate events without leaking
// credentials into logs.
type AuditEvent struct {
	Timestamp        time.Time
	AccessorID       string
	Namespace        string
	Decision         string // "resolved" or "denied"
	Reason           string
	PolicyVersion    uint64
	CacheHit         bool
	TokenFingerprint string
}

// AuditSink receives an AuditEvent for every resolution attempt.
type AuditSink interface {
	Emit(AuditEvent)
}

// AuditFunc adapts a function to an AuditSink.
type AuditFunc func(AuditEvent)

// Emit implements AuditSink.
func (f AuditFunc) Emit(e AuditEvent) { f(e) }

// tokenFingerprint returns a non-reversible, truncated identifier for a
// secret, safe to put in audit logs.
func tokenFingerprint(secret string) string {
	sum := sha256.Sum256([]byte(secret))
	return hex.EncodeToString(sum[:])[:12]
}

// LocalResult is the outcome of a successful LocalResolver.Resolve.
type LocalResult struct {
	Authorizer    acl.Authorizer
	AccessorID    string
	Namespace     string
	PolicyVersion uint64
}

// localCacheKey binds a compiled authorizer to both the token identity and
// the exact policy generation it was compiled from. Omitting either half
// would let one token inherit another token's permissions, or let a token
// keep permissions from a stale policy set.
type localCacheKey struct {
	accessorID string
	version    uint64
}

// LocalResolver resolves (namespace, secret) pairs to authorizers against a
// LocalStore, caching compiled authorizers per (identity, generation).
type LocalResolver struct {
	store  *LocalStore
	conf   *acl.Config
	audit  AuditSink
	now    func() time.Time
	sf     singleflight.Group

	mu    sync.RWMutex
	cache map[localCacheKey]acl.Authorizer
}

// LocalResolverConfig configures a LocalResolver.
type LocalResolverConfig struct {
	Store  *LocalStore
	Config *acl.Config
	Audit  AuditSink
	// Now overrides the clock, mainly for tests. Defaults to time.Now.
	Now func() time.Time
}

func NewLocalResolver(cfg LocalResolverConfig) (*LocalResolver, error) {
	if cfg.Store == nil {
		return nil, fmt.Errorf("LocalResolver requires a LocalStore")
	}
	if cfg.Audit == nil {
		cfg.Audit = AuditFunc(func(AuditEvent) {})
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &LocalResolver{
		store:  cfg.Store,
		conf:   cfg.Config,
		audit:  cfg.Audit,
		now:    cfg.Now,
		cache:  make(map[localCacheKey]acl.Authorizer),
	}, nil
}

func (r *LocalResolver) emit(e AuditEvent) {
	e.Timestamp = r.now()
	r.audit.Emit(e)
}

// Resolve maps a secret within a namespace to an authorizer.
//
// Token validity (existence, revocation, expiry) is always evaluated
// against the store's current generation before any cache entry is
// consulted, so a revoked or expired token can never be served a stale
// cached result. The compiled authorizer is cached under
// (accessor ID, generation): any policy load, token update, revocation or
// rollback changes the generation and makes older entries unreachable.
func (r *LocalResolver) Resolve(namespace, secret string) (*LocalResult, error) {
	namespace = normalizeNamespace(namespace)
	fp := tokenFingerprint(secret)

	deny := func(reason string, err error) (*LocalResult, error) {
		r.emit(AuditEvent{
			Namespace:        namespace,
			Decision:         "denied",
			Reason:           reason,
			PolicyVersion:    r.store.Version(),
			TokenFingerprint: fp,
		})
		return nil, err
	}

	if secret == "" {
		return deny("empty-secret", acl.ErrNotFound)
	}
	// Reserved root identifiers never resolve to a policy set.
	if acl.RootAuthorizer(secret) != nil {
		return deny("root-token", acl.ErrRootDenied)
	}

	snap := r.store.snapshot()

	accessorID, ok := snap.secrets[nsKey(namespace, secret)]
	if !ok {
		return deny("token-not-found", acl.ErrNotFound)
	}
	tok := snap.tokens[nsKey(namespace, accessorID)]
	if tok == nil {
		return deny("token-not-found", acl.ErrNotFound)
	}
	if tok.expired(r.now()) {
		return deny("token-expired", acl.ErrNotFound)
	}

	key := localCacheKey{accessorID: accessorID, version: snap.version}

	r.mu.RLock()
	cached, hit := r.cache[key]
	r.mu.RUnlock()
	if hit {
		r.emit(AuditEvent{
			AccessorID:       accessorID,
			Namespace:        namespace,
			Decision:         "resolved",
			Reason:           "cache-hit",
			PolicyVersion:    snap.version,
			CacheHit:         true,
			TokenFingerprint: fp,
		})
		return &LocalResult{
			Authorizer:    cached,
			AccessorID:    accessorID,
			Namespace:     namespace,
			PolicyVersion: snap.version,
		}, nil
	}

	// Compile at most once per (identity, generation) even under concurrent
	// resolution. The singleflight function reads only from the captured
	// immutable snapshot, so the resulting authorizer always reflects one
	// complete, consistent policy set.
	v, err, _ := r.sf.Do(fmt.Sprintf("%s\x00%d", accessorID, snap.version), func() (interface{}, error) {
		// Double-check after winning/awaiting the flight.
		r.mu.RLock()
		authz, ok := r.cache[key]
		r.mu.RUnlock()
		if ok {
			return authz, nil
		}

		var parsed []*acl.Policy
		for _, name := range tok.Policies {
			p, ok := snap.policies[namespace][name]
			if !ok {
				// A policy referenced by the token is gone from this
				// generation; skip it rather than failing the whole
				// resolution, matching server-side resolution semantics.
				continue
			}
			parsed = append(parsed, p)
		}

		authz, err := acl.NewPolicyAuthorizerWithDefaults(acl.DenyAll(), parsed, r.conf)
		if err != nil {
			return nil, err
		}

		r.putCache(key, authz)
		return authz, nil
	})
	if err != nil {
		return deny("compile-error", err)
	}

	r.emit(AuditEvent{
		AccessorID:       accessorID,
		Namespace:        namespace,
		Decision:         "resolved",
		Reason:           "compiled",
		PolicyVersion:    snap.version,
		TokenFingerprint: fp,
	})
	return &LocalResult{
		Authorizer:    v.(acl.Authorizer),
		AccessorID:    accessorID,
		Namespace:     namespace,
		PolicyVersion: snap.version,
	}, nil
}

// putCache stores an authorizer and evicts entries from generations that are
// neither current nor the rollback target, bounding the cache to two
// generations.
func (r *LocalResolver) putCache(key localCacheKey, authz acl.Authorizer) {
	cur, prev := r.store.Versions()

	r.mu.Lock()
	defer r.mu.Unlock()
	for k := range r.cache {
		if k.version != cur && k.version != prev {
			delete(r.cache, k)
		}
	}
	r.cache[key] = authz
}

// CacheSize reports the number of cached authorizers; used by tests.
func (r *LocalResolver) CacheSize() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.cache)
}
