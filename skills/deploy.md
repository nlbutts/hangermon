# Deploy Skill

This skill provides a command to deploy the hangermon code to a target unit.

## Usage

```
/deploy [unit]
```

## Parameters

- `unit` (optional): The target unit name. Defaults to "hanger".

## Examples

```
/deploy hanger
/deploy hangermon2
```

## Implementation

This skill runs the `sync.sh` script to rsync code to the specified unit.
