// Copyright IBM Corp. 2024, 2026
// SPDX-License-Identifier: BUSL-1.1

package resolver

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hashicorp/consul/acl"
)

func mustLoad(t *testing.T, s *LocalStore, ns string, texts map[string]string) uint64 {
	t.Helper()
	v, err := s.LoadPolicies(ns, texts, nil)
	if err != nil {
		t.Fatalf("LoadPolicies: %v", err)
	}
	return v
}

func mustUpsert(t *testing.T, s *LocalStore, tok LocalToken) {
	t.Helper()
	if _, err := s.UpsertToken(tok); err != nil {
		t.Fatalf("UpsertToken: %v", err)
	}
}

func mustResolve(t *testing.T, r *LocalResolver, ns, secret string) *LocalResult {
	t.Helper()
	res, err := r.Resolve(ns, secret)
	if err != nil {
		t.Fatalf("Resolve(%q, ...): %v", ns, err)
	}
	return res
}

// TestEvaluationBoundaries pins the evaluation semantics that must remain
// stable across hot reloads: path separation, raw prefix rules, exact-match
// precedence and deny priority.
func TestEvaluationBoundaries(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"main": `
key_prefix "foo/"   { policy = "write" }
key_prefix "foo/ba" { policy = "read" }
key "foo/bar"       { policy = "deny" }
key_prefix "raw"    { policy = "read" }
`,
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "secret-1", Policies: []string{"main"},
	})

	r, err := NewLocalResolver(LocalResolverConfig{Store: store})
	if err != nil {
		t.Fatal(err)
	}
	res := mustResolve(t, r, "default", "secret-1")
	authz := res.Authorizer
	ctx := &acl.AuthorizerContext{}

	check := func(key string, read, write acl.EnforcementDecision) {
		t.Helper()
		if got := authz.KeyRead(key, ctx); got != read {
			t.Errorf("KeyRead(%q) = %v, want %v", key, got, read)
		}
		if got := authz.KeyWrite(key, ctx); got != write {
			t.Errorf("KeyWrite(%q) = %v, want %v", key, got, write)
		}
	}

	// Exact deny beats every prefix rule on the same path.
	check("foo/bar", acl.Deny, acl.Deny)
	// Longest prefix wins: "foo/ba" (read) outranks "foo/" (write).
	check("foo/baz", acl.Allow, acl.Deny)
	// Plain prefix rule applies below "foo/".
	check("foo/other", acl.Allow, acl.Allow)
	// Path separation: "foo/" must NOT match the adjacent path "foobar".
	check("foobar", acl.Deny, acl.Deny)
	// A raw prefix without a separator ("raw") is a pure string prefix and
	// does match "rawhide" — documenting prefix semantics explicitly.
	check("rawhide", acl.Allow, acl.Deny)
	// Unrelated paths fall through to the default deny.
	check("elsewhere", acl.Deny, acl.Deny)
}

// TestDenyPrecedenceAcrossPolicies verifies deny stays dominant when rules
// for the same path come from different policies of one token.
func TestDenyPrecedenceAcrossPolicies(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"writer": `key_prefix "svc/" { policy = "write" }`,
		"guard":  `key_prefix "svc/" { policy = "deny" }`,
		"reader": `key_prefix "ro/"  { policy = "read" }`,
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "secret-1",
		Policies: []string{"writer", "guard", "reader"},
	})

	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})
	res := mustResolve(t, r, "default", "secret-1")
	ctx := &acl.AuthorizerContext{}

	if got := res.Authorizer.KeyWrite("svc/x", ctx); got != acl.Deny {
		t.Fatalf("deny must override write for the same prefix, got %v", got)
	}
	if got := res.Authorizer.KeyRead("svc/x", ctx); got != acl.Deny {
		t.Fatalf("deny must override read for the same prefix, got %v", got)
	}
	// read grants read but not list/write.
	if got := res.Authorizer.KeyRead("ro/x", ctx); got != acl.Allow {
		t.Fatalf("read policy must allow reads, got %v", got)
	}
	if got := res.Authorizer.KeyWrite("ro/x", ctx); got != acl.Deny {
		t.Fatalf("read policy must not allow writes, got %v", got)
	}
}

// TestHotReloadAtomicity hammers Resolve while policy generations flip back
// and forth. Every single observation must be one complete generation:
// generation A allows x and denies y, generation B denies x and allows y.
// A torn read (both allowed or both denied) means a half-applied policy set.
func TestHotReloadAtomicity(t *testing.T) {
	store := NewLocalStore()
	genA := map[string]string{"p": `
key "x" { policy = "read" }
key "y" { policy = "deny" }
`}
	genB := map[string]string{"p": `
key "x" { policy = "deny" }
key "y" { policy = "read" }
`}
	mustLoad(t, store, "default", genA)
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "secret-1", Policies: []string{"p"},
	})

	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})
	ctx := &acl.AuthorizerContext{}

	var torn atomic.Int64
	var stop atomic.Bool
	var wg sync.WaitGroup

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stop.Load() {
				res, err := r.Resolve("default", "secret-1")
				if err != nil {
					continue
				}
				x := res.Authorizer.KeyRead("x", ctx) == acl.Allow
				y := res.Authorizer.KeyRead("y", ctx) == acl.Allow
				if x == y {
					torn.Add(1)
				}
			}
		}()
	}

	for i := 0; i < 200; i++ {
		if i%2 == 0 {
			mustLoad(t, store, "default", genB)
		} else {
			mustLoad(t, store, "default", genA)
		}
	}
	stop.Store(true)
	wg.Wait()

	if n := torn.Load(); n != 0 {
		t.Fatalf("observed %d torn policy sets during hot reload", n)
	}
}

// TestRevokedTokenGetsNoStaleResult ensures revoking a token immediately
// invalidates previously cached authorizations for its secret.
func TestRevokedTokenGetsNoStaleResult(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"p": `key "k" { policy = "read" }`,
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "secret-1", Policies: []string{"p"},
	})
	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})

	res := mustResolve(t, r, "default", "secret-1")
	if got := res.Authorizer.KeyRead("k", nil); got != acl.Allow {
		t.Fatalf("pre-revocation read = %v, want Allow", got)
	}
	// Second resolve comes from the cache.
	res = mustResolve(t, r, "default", "secret-1")
	if got := res.Authorizer.KeyRead("k", nil); got != acl.Allow {
		t.Fatalf("cached read = %v, want Allow", got)
	}

	if _, ok := store.RevokeToken("default", "acc-1"); !ok {
		t.Fatal("expected token to exist")
	}
	if _, err := r.Resolve("default", "secret-1"); !acl.IsErrNotFound(err) {
		t.Fatalf("revoked token resolved: err=%v", err)
	}
}

// TestExpiredTokenGetsNoStaleResult uses a controllable clock to show that
// an expired token cannot be served from the cache.
func TestExpiredTokenGetsNoStaleResult(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"p": `key "k" { policy = "read" }`,
	})

	now := time.Now()
	clock := &now
	r, _ := NewLocalResolver(LocalResolverConfig{
		Store: store,
		Now:   func() time.Time { return *clock },
	})

	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "secret-1", Policies: []string{"p"},
		ExpiresAt: now.Add(time.Minute),
	})
	if _, err := r.Resolve("default", "secret-1"); err != nil {
		t.Fatalf("unexpired token should resolve: %v", err)
	}

	*clock = now.Add(2 * time.Minute)
	if _, err := r.Resolve("default", "secret-1"); !acl.IsErrNotFound(err) {
		t.Fatalf("expired token resolved from cache: err=%v", err)
	}
}

// TestConcurrentReadWrite exercises loads, token upserts, revocations and
// resolves concurrently. Run with -race; any resolved result must be
// internally consistent with a single generation.
func TestConcurrentReadWrite(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"p": `key "k" { policy = "read" }`,
	})
	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})
	ctx := &acl.AuthorizerContext{}

	var stop atomic.Bool
	var readers, writers sync.WaitGroup

	// Readers: several tokens, some present, some revoked mid-flight.
	for i := 0; i < 8; i++ {
		i := i
		readers.Add(1)
		go func() {
			defer readers.Done()
			secret := fmt.Sprintf("secret-%d", i%4)
			for !stop.Load() {
				res, err := r.Resolve("default", secret)
				if err != nil {
					continue
				}
				// The only policy ever loaded grants read on "k" and
				// nothing else; any other decision is corruption.
				if res.Authorizer.KeyRead("k", ctx) != acl.Allow ||
					res.Authorizer.KeyWrite("k", ctx) != acl.Deny {
					t.Errorf("impossible decision for %q", secret)
					return
				}
			}
		}()
	}

	// Writers: churn tokens and reload equivalent policies.
	for i := 0; i < 4; i++ {
		i := i
		writers.Add(1)
		go func() {
			defer writers.Done()
			for n := 0; n < 100; n++ {
				tok := LocalToken{
					AccessorID: fmt.Sprintf("acc-%d", n%4),
					SecretID:   fmt.Sprintf("secret-%d", n%4),
					Policies:   []string{"p"},
				}
				_, _ = store.UpsertToken(tok)
				if i%2 == 0 {
					_, _ = store.LoadPolicies("default", map[string]string{
						"p": `key "k" { policy = "read" }`,
					}, nil)
				}
				if n%3 == 0 {
					_, _ = store.RevokeToken("default", fmt.Sprintf("acc-%d", n%4))
				}
			}
		}()
	}

	writers.Wait()
	stop.Store(true)
	readers.Wait()
}

// TestVersionRollback verifies rollback restores the exact previous
// generation and that no cached result from the rolled-forward generation
// leaks into post-rollback decisions.
func TestVersionRollback(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"p": `key "k" { policy = "read" }`,
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "secret-1", Policies: []string{"p"},
	})
	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})

	res := mustResolve(t, r, "default", "secret-1")
	if got := res.Authorizer.KeyRead("k", nil); got != acl.Allow {
		t.Fatalf("v1 read = %v, want Allow", got)
	}
	v1 := res.PolicyVersion

	// Roll forward to a generation that denies the key.
	mustLoad(t, store, "default", map[string]string{
		"p": `key "k" { policy = "deny" }`,
	})
	res = mustResolve(t, r, "default", "secret-1")
	if got := res.Authorizer.KeyRead("k", nil); got != acl.Deny {
		t.Fatalf("v2 read = %v, want Deny", got)
	}
	if res.PolicyVersion == v1 {
		t.Fatal("version did not advance after reload")
	}

	restored, ok := store.Rollback()
	if !ok {
		t.Fatal("expected a rollback target")
	}
	if restored != v1 {
		t.Fatalf("rollback restored version %d, want %d", restored, v1)
	}

	res = mustResolve(t, r, "default", "secret-1")
	if got := res.Authorizer.KeyRead("k", nil); got != acl.Allow {
		t.Fatalf("post-rollback read = %v, want Allow (v1 semantics)", got)
	}
	if res.PolicyVersion != v1 {
		t.Fatalf("post-rollback version = %d, want %d", res.PolicyVersion, v1)
	}
}

// TestCacheKeyIdentityIsolation reproduces the "authorization bleed"
// failure mode: two tokens with different permissions resolved
// back-to-back must never share a cached authorizer.
func TestCacheKeyIdentityIsolation(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"allow-k": `key "k" { policy = "read" }`,
		"deny-k":  `key "k" { policy = "deny" }`,
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-allow", SecretID: "secret-allow", Policies: []string{"allow-k"},
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-deny", SecretID: "secret-deny", Policies: []string{"deny-k"},
	})
	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})

	for i := 0; i < 50; i++ {
		ra := mustResolve(t, r, "default", "secret-allow")
		if got := ra.Authorizer.KeyRead("k", nil); got != acl.Allow {
			t.Fatalf("iteration %d: allow-token got %v", i, got)
		}
		rb := mustResolve(t, r, "default", "secret-deny")
		if got := rb.Authorizer.KeyRead("k", nil); got != acl.Deny {
			t.Fatalf("iteration %d: deny-token inherited allow decision (%v)", i, got)
		}
	}

	// After a hot reload that swaps the policy contents, each token must
	// still get exactly its own (new) permissions.
	mustLoad(t, store, "default", map[string]string{
		"allow-k": `key "k" { policy = "deny" }`,
		"deny-k":  `key "k" { policy = "read" }`,
	})
	ra := mustResolve(t, r, "default", "secret-allow")
	if got := ra.Authorizer.KeyRead("k", nil); got != acl.Deny {
		t.Fatalf("post-reload allow-token got %v, want Deny", got)
	}
	rb := mustResolve(t, r, "default", "secret-deny")
	if got := rb.Authorizer.KeyRead("k", nil); got != acl.Allow {
		t.Fatalf("post-reload deny-token got %v, want Allow", got)
	}
}

// TestNamespaceIsolation ensures secrets, policies and cache entries never
// cross namespace boundaries.
func TestNamespaceIsolation(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "team-a", map[string]string{
		"p": `key "k" { policy = "read" }`,
	})
	mustLoad(t, store, "team-b", map[string]string{
		"p": `key "k" { policy = "deny" }`,
	})
	// Same secret string and same policy name in both namespaces.
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-a", SecretID: "shared-secret", Namespace: "team-a", Policies: []string{"p"},
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-b", SecretID: "shared-secret", Namespace: "team-b", Policies: []string{"p"},
	})
	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})

	ra := mustResolve(t, r, "team-a", "shared-secret")
	if got := ra.Authorizer.KeyRead("k", nil); got != acl.Allow {
		t.Fatalf("team-a got %v, want Allow", got)
	}
	rb := mustResolve(t, r, "team-b", "shared-secret")
	if got := rb.Authorizer.KeyRead("k", nil); got != acl.Deny {
		t.Fatalf("team-b got %v, want Deny", got)
	}

	// A token only exists in its own namespace.
	if _, err := r.Resolve("team-c", "shared-secret"); !acl.IsErrNotFound(err) {
		t.Fatalf("secret leaked into team-c: err=%v", err)
	}
}

// TestAuditEventsNeverContainSecrets performs resolutions and denials and
// scans every rendered audit event for token material.
func TestAuditEventsNeverContainSecrets(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"p": `key "k" { policy = "read" }`,
	})
	const secret = "super-secret-token-value-12345"
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: secret, Policies: []string{"p"},
	})

	var mu sync.Mutex
	var events []AuditEvent
	r, _ := NewLocalResolver(LocalResolverConfig{
		Store: store,
		Audit: AuditFunc(func(e AuditEvent) {
			mu.Lock()
			events = append(events, e)
			mu.Unlock()
		}),
	})

	mustResolve(t, r, "default", secret) // compiled
	mustResolve(t, r, "default", secret) // cache hit
	_, _ = r.Resolve("default", "another-secret-never-logged")
	_, _ = r.Resolve("default", "")

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 4 {
		t.Fatalf("expected 4 audit events, got %d", len(events))
	}
	secrets := []string{secret, "another-secret-never-logged"}
	for _, e := range events {
		rendered := fmt.Sprintf("%+v", e)
		for _, s := range secrets {
			if strings.Contains(rendered, s) {
				t.Fatalf("audit event leaks token material: %s", rendered)
			}
		}
	}
	// The resolved events must carry identity and version for forensics.
	var hits, compiled int
	for _, e := range events {
		if e.Decision != "resolved" {
			continue
		}
		if e.AccessorID != "acc-1" {
			t.Fatalf("resolved event missing accessor: %+v", e)
		}
		if e.PolicyVersion == 0 {
			t.Fatalf("resolved event missing policy version: %+v", e)
		}
		if e.CacheHit {
			hits++
		} else {
			compiled++
		}
	}
	if hits != 1 || compiled != 1 {
		t.Fatalf("expected 1 compiled + 1 cache-hit event, got %d/%d", compiled, hits)
	}
}

// TestFailedLoadKeepsPreviousGeneration proves a parse error is
// all-or-nothing: the store keeps serving the last good generation.
func TestFailedLoadKeepsPreviousGeneration(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"p": `key "k" { policy = "read" }`,
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "secret-1", Policies: []string{"p"},
	})
	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})
	mustResolve(t, r, "default", "secret-1")

	before := store.Version()
	_, err := store.LoadPolicies("default", map[string]string{
		"good": `key "k" { policy = "deny" }`,
		"bad":  `key "k" { policy = "not-a-level" }`,
	}, nil)
	if err == nil {
		t.Fatal("expected parse error")
	}
	if got := store.Version(); got != before {
		t.Fatalf("failed load changed generation: before=%d after=%d", before, got)
	}

	res := mustResolve(t, r, "default", "secret-1")
	if got := res.Authorizer.KeyRead("k", nil); got != acl.Allow {
		t.Fatalf("failed load changed decisions: got %v, want Allow", got)
	}
}

// TestSecretRebindingInvalidatesOldSecret covers rotating a token's secret:
// the old secret must stop resolving immediately.
func TestSecretRebindingInvalidatesOldSecret(t *testing.T) {
	store := NewLocalStore()
	mustLoad(t, store, "default", map[string]string{
		"p": `key "k" { policy = "read" }`,
	})
	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "old-secret", Policies: []string{"p"},
	})
	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})
	mustResolve(t, r, "default", "old-secret")

	mustUpsert(t, store, LocalToken{
		AccessorID: "acc-1", SecretID: "new-secret", Policies: []string{"p"},
	})
	if _, err := r.Resolve("default", "old-secret"); !acl.IsErrNotFound(err) {
		t.Fatalf("old secret still resolves: err=%v", err)
	}
	res := mustResolve(t, r, "default", "new-secret")
	if got := res.Authorizer.KeyRead("k", nil); got != acl.Allow {
		t.Fatalf("new secret read = %v, want Allow", got)
	}
}

// TestRootAndEmptySecretsRejected pins the reserved-token boundary.
func TestRootAndEmptySecretsRejected(t *testing.T) {
	store := NewLocalStore()
	r, _ := NewLocalResolver(LocalResolverConfig{Store: store})

	if _, err := r.Resolve("default", ""); !acl.IsErrNotFound(err) {
		t.Fatalf("empty secret: err=%v", err)
	}
	if _, err := r.Resolve("default", "allow"); !acl.IsErrRootDenied(err) {
		t.Fatalf("root identifier: err=%v", err)
	}
}
