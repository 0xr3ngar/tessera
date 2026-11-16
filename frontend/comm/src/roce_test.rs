use crate::{port, ports};
use engine_comm::Error;

#[test]
fn specs_name_a_gid() {
    let p = port("mlx5_0:3").unwrap();
    assert_eq!((p.device.as_str(), p.gid), ("mlx5_0", 3));
    let two = ports("a:1,b:2").unwrap();
    assert_eq!(two.len(), 2);
    assert_eq!(two[1].gid, 2);
    assert!(matches!(port("a:x"), Err(Error::Fit(_))));
    // No such device, so the sysfs lookup fails.
    assert!(port("no-such-device").is_err());
}
