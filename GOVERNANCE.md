# Governance

This document describes the governance model for the Tern project.

## Principles

- **Community-first**: Decisions are made by the community, for the community
- **Transparency**: All discussions and decisions happen in public
- **Inclusivity**: Everyone is welcome to participate
- **Sustainability**: The project should outlive any single contributor

## Roles

### Maintainers

Maintainers have write access to the repository and are responsible for:
- Reviewing and merging pull requests
- Managing releases
- Setting project direction
- Enforcing code of conduct

Maintainers are selected based on sustained, high-quality contributions.

### Contributors

Anyone who contributes code, documentation, bug reports, or other value to the project.

## Decision Making

### Consensus-Based

Most decisions are made by consensus among maintainers. If consensus cannot be reached:
1. Discussion continues for a reasonable period (minimum 7 days)
2. If still unresolved, a vote among maintainers decides
3. Simple majority wins; maintainer who proposed the change abstains

### Types of Decisions

| Decision Type | Who Decides | Process |
|---|---|---|
| Bug fixes | Any maintainer | Merge after review |
| New features | Maintainers | Consensus or vote |
| Breaking changes | Maintainers | Consensus required |
| Governance changes | All maintainers | Supermajority (2/3) |
| Adding maintainers | Maintainers | Consensus required |
| Removing maintainers | Maintainers | Supermajority (2/3) |

## Adding Maintainers

1. Nomination by existing maintainer
2. Public discussion (minimum 7 days)
3. Consensus among maintainers
4. Invitation to nominee

Criteria for maintainership:
- Sustained contributions over 3+ months
- Demonstrated understanding of project goals
- Good communication and collaboration skills
- Commitment to project values

## Removing Maintainers

Maintainers may step down at any time.

Involuntary removal requires:
- Supermajority vote (2/3) of other maintainers
- Documented reasons (inactivity, code of conduct violations, etc.)
- Opportunity for the maintainer to respond

## Releases

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Release Process

1. Maintainer proposes release
2. Changelog generated from PRs since last release
3. Version bumped
4. Release candidate published for testing (minimum 3 days)
5. Final release tagged and published

## Code of Conduct Enforcement

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for enforcement procedures.

## License

All contributions are licensed under Apache-2.0. See [LICENSE](LICENSE).

## Amendments

This governance document can be amended by supermajority vote of maintainers.