// Commitlint: Conventional Commits. Kept in sync with docs/commit-convention.md
// @see https://github.com/conventional-changelog/commitlint
/** @type {import("@commitlint/types").UserConfig} */
module.exports = {
  extends: ["@commitlint/config-conventional"],
  defaultIgnores: true,
  helpUrl:
    "https://github.com/cesar-andress/fightsafe-ai/blob/main/docs/commit-convention.md",
  rules: {
    "header-max-length": [2, "always", 72],
    "body-max-line-length": [0],
    "footer-max-line-length": [0],
    "type-case": [2, "always", "lower-case"],
    "type-enum": [
      2,
      "always",
      [
        "build",
        "chore",
        "ci",
        "docs",
        "feat",
        "fix",
        "perf",
        "refactor",
        "revert",
        "style",
        "test",
      ],
    ],
  },
};
