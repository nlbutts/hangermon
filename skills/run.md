# Run Skill

This skill provides a command to run the hangermon project on a remote unit.

## Usage

```
/run [unit]
```

## Parameters

- `unit` (optional): The target unit name. Defaults to "hanger".

## Examples

```
/run hanger
/run hangermon2
```

## Implementation

This skill runs the following command on the remote unit:
```bash
ssh <user>@<unit> 'cd /home/nlbutts/hangermon && uv run app.py'
```

The command is executed in the background and detached from the SSH session.
